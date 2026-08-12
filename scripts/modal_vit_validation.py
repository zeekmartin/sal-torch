"""Does SAL work on vision transformers under full fine-tuning?

Every multi-seed result this project has is GPT-2 Medium on SST-2 — one
architecture, one modality, one task. SAL's mechanism has nothing text-specific
about it (silence attention heads during training, make the model reorganize),
so a vision transformer is the cheapest real test of whether the effect is a
property of transformers or a property of *that* run.

There is also a specific claim to check. The original research reported a **+20pp
gain on ViT** under compression. Nothing in this package has ever reproduced
that, and a +20pp effect is large enough that failing to see it means something
either way.

Protocol
--------
ViT-base-patch16-224 on CIFAR-10, **full fine-tuning** (every parameter
trainable — never LoRA, which the v0.4.0 work established starves the
mechanism). Three seeds: 42, 123, 456.

Per seed, both arms train from the same ImageNet-pretrained checkpoint:

* **baseline** — 3 epochs, AdamW, lr=3e-5, batch 32, gradient clipping at 1.0.
* **SAL** — identical, plus ``HeadMasker`` at ``prune_fraction=0.33``.

Then five compression variants per arm::

    dense, prune33, prune50, int8, prune33+int8

``int8`` is ``torch.ao`` dynamic quantization, which is CPU-only, so those
variants are scored on CPU (the classifier head is never quantized). Pruning
masks randomly chosen heads — ``random`` is the shipped default strategy.
Combined variants quantize first, then install the mask, so the hooks land on
the quantized modules rather than being deep-copied through them.

As in the GPT-2 multi-seed run, the eval-time head choice is seeded separately
from the training seed (``10000 + seed``), so SAL is never scored on exactly the
heads it trained against. Both arms of a seed lose the same heads.

Reading the numbers
-------------------
Three seeds is thin. A variant is called consistent only when SAL leads on
**all three** — with n=3 there is no meaningful middle ground, and 2/3 is a coin
flip dressed up. Mean ± std is reported for scale, not for significance.

Note the ceiling: ViT-base is ImageNet-pretrained and CIFAR-10 is easy, so the
dense arms land high and there is little headroom for SAL to show a *clean*
accuracy gain. The compressed rows are where the question actually lives.

Usage::

    modal run scripts/modal_vit_validation.py             # the real run
    SAL_SMOKE=1 modal run scripts/modal_vit_validation.py # plumbing check

Environment overrides: ``SAL_SEEDS``, ``SAL_N_TRAIN``, ``SAL_N_EVAL``,
``SAL_EPOCHS``. Results are written to ``scripts/vit_validation_results.json``.
"""
from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("sal-torch-vit-validation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers", "datasets", "numpy",
                 "accelerate>=1.1.0", "pillow")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

VARIANTS = ["dense", "prune33", "prune50", "int8", "prune33+int8"]

# Passed to the remote function as an argument, never read from os.environ inside
# the container: `modal run` does not forward the local environment, so a
# container reading env vars would silently ignore every override.
BASE_CONFIG = {
    "model": "google/vit-base-patch16-224",
    "task": "cifar10",
    "n_train": 2048,
    "n_eval": 1000,
    "epochs": 3,
    "lr": 3e-5,
    "train_bs": 32,
    "eval_bs": 64,
    "prune_fraction": 0.33,
    "battery_seed_offset": 10_000,
}


def _build_config():
    """(seeds, config) for this invocation, from the optional env overrides."""
    smoke = os.environ.get("SAL_SMOKE", "").strip() not in ("", "0", "false")
    seeds = [int(s) for s in
             os.environ.get("SAL_SEEDS", "42,123,456").split(",") if s.strip()]
    cfg = dict(BASE_CONFIG)
    for key, env in (("n_train", "SAL_N_TRAIN"), ("n_eval", "SAL_N_EVAL"),
                     ("epochs", "SAL_EPOCHS")):
        if os.environ.get(env):
            cfg[key] = int(os.environ[env])
    if smoke:
        seeds = seeds[:1]
        cfg.update(n_train=64, n_eval=64, epochs=1)
    cfg["smoke"] = smoke
    return seeds, cfg


SEEDS, CONFIG = _build_config()
RESULTS_PATH = ("scripts/vit_smoke.json" if CONFIG["smoke"]
                else "scripts/vit_validation_results.json")


# ------------------------------------------------------------------ data loading
def load_cifar10(model_name: str, n_train: int, n_eval: int, seed: int):
    """(train_batches, eval_batches) as lists of {pixel_values, labels}.

    Images are preprocessed once and cached as tensors — 3 epochs over the same
    2048 images would otherwise re-run the PIL resize nine times for nothing.
    The eval split is held fixed across seeds so every seed is scored on exactly
    the same images; only the training subset is reshuffled.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_name)
    ds = load_dataset("uoft-cs/cifar10")

    def to_batches(split, bs: int):
        out = []
        imgs, labels = split["img"], split["label"]
        for i in range(0, len(imgs), bs):
            px = proc(images=imgs[i:i + bs], return_tensors="pt")["pixel_values"]
            out.append({"pixel_values": px,
                        "labels": torch.tensor(labels[i:i + bs], dtype=torch.long)})
        return out

    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    ev = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    return to_batches(train, 32), to_batches(ev, 64)


# -------------------------------------------------------------------- evaluation
def evaluate_accuracy(model, batches, device) -> float:
    """Top-1 accuracy over pre-built batches."""
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in batches:
            px = b["pixel_values"].to(device)
            labels = b["labels"].to(device)
            logits = model(pixel_values=px).logits
            correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.shape[0]
    return correct / max(total, 1)


# --------------------------------------------------------------------- training
def train_arm(model, batches, device, use_sal: bool, seed: int, cfg: dict):
    """Fully fine-tune one arm, optionally with SAL head masking.

    The masker is installed after ``.to(device)`` so its masks are allocated
    where the model lives, and removed before returning so the compression
    battery sees a plain model whose weights already carry the adaptation.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    from sal.config import SALConfig
    from sal.masker import HeadMasker

    epochs = cfg["epochs"]

    torch.manual_seed(seed)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    masker = None
    if use_sal:
        config = SALConfig.auto(model, prune_fraction=cfg["prune_fraction"])
        masker = HeadMasker(model, config, seed=seed)
        masker.install()
        print(f"    SAL: {config.num_layers} layers x {config.num_heads_per_layer} "
              f"heads, target {config.num_heads_to_prune} silenced", flush=True)

    total_steps = max(1, len(batches) * epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"    full FT: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M trainable "
          f"({n_trainable / max(n_total, 1):.1%}), lr={cfg['lr']:g}, {total_steps} steps",
          flush=True)

    opt = AdamW(params, lr=cfg["lr"])
    model.train()

    rng = np.random.RandomState(seed)
    step = 0
    for _ in range(epochs):
        for idx in rng.permutation(len(batches)):
            if masker is not None:
                masker.step(step, total_steps)
            b = batches[idx]
            loss = model(pixel_values=b["pixel_values"].to(device),
                         labels=b["labels"].to(device)).loss
            loss.backward()
            # Silencing a third of the heads makes gradients spikier than
            # ordinary fine-tuning; SALTrainer clips by default, so this does too.
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

    if masker is not None:
        stats = masker.stats
        masker.remove()
        print(f"    SAL: {stats['pruned_heads']}/{stats['total_heads']} heads silenced "
              f"over {stats['prune_events']} prune events", flush=True)

    model.eval()
    return model


# ----------------------------------------------------------------- the battery
def _build_variant(master, variant: str, device, battery_seed: int):
    """Materialize one compression variant. Returns (model, backend, device, masker)."""
    import copy

    from sal.config import SALConfig
    from sal.masker import HeadMasker
    from sal.quantize import quantize

    model = copy.deepcopy(master)
    backend, dev = None, device

    if "int8" in variant:
        # torch.ao dynamic INT8 is CPU-only, so this variant is scored on CPU.
        model = quantize(model, method="int8", backend="torch_ao", inplace=True)
        backend, dev = "torch-dynamic-int8", "cpu"
    else:
        model = model.to(device)

    masker = None
    if variant.startswith("prune"):
        pct = int(variant.split("+")[0].replace("prune", ""))
        config = SALConfig.auto(model, prune_fraction=pct / 100.0)
        masker = HeadMasker(model, config, seed=battery_seed)
        masker.install()
        masker.activate()      # random selection — the shipped default strategy

    return model, backend, dev, masker


def run_battery(master, batches, device, battery_seed: int, label: str) -> dict:
    """Evaluate every compression variant of one trained model."""
    import torch

    from sal.quantize import model_size_mb

    out = {}
    for variant in VARIANTS:
        t0 = time.time()
        model = masker = None
        try:
            model, backend, dev, masker = _build_variant(master, variant, device,
                                                         battery_seed)
            acc = evaluate_accuracy(model, batches, dev)
            size = model_size_mb(model)
            out[variant] = {"accuracy": acc, "size_mb": round(size, 1),
                            "backend": backend, "device": dev,
                            "seconds": round(time.time() - t0, 1)}
            print(f"  [{label}] {variant:<14} acc={acc:.4f}  size={size:7.1f}MB  "
                  f"({dev}{'/' + backend if backend else ''})  "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001 — a dead variant shouldn't sink the seed
            out[variant] = {"accuracy": None, "error": str(e),
                            "seconds": round(time.time() - t0, 1)}
            print(f"  [{label}] {variant:<14} FAILED: {e}", flush=True)
        finally:
            if masker is not None:
                masker.remove()
            del model, masker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


# -------------------------------------------------------------------- one seed
@app.function(image=image, gpu="T4", cpu=16.0, memory=32768, timeout=1800)
def run_seed(seed: int, cfg: dict) -> dict:
    """Train both arms at this seed and score the full battery on each."""
    import torch
    from transformers import ViTForImageClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 8)     # the int8 variants run here
    print(f"=== seed {seed}: {cfg['model']} / {cfg['task']}, full fine-tuning "
          f"on {device} ===", flush=True)

    t0 = time.time()
    train_b, eval_b = load_cifar10(cfg["model"], cfg["n_train"], cfg["n_eval"], seed)
    n_train = sum(b["labels"].shape[0] for b in train_b)
    n_eval = sum(b["labels"].shape[0] for b in eval_b)
    print(f"    {n_train} train / {n_eval} eval images preprocessed in "
          f"{time.time() - t0:.0f}s, {cfg['epochs']} epochs, lr={cfg['lr']:g}, "
          f"batch {cfg['train_bs']}", flush=True)

    battery_seed = cfg["battery_seed_offset"] + seed
    results = {}
    for arm, use_sal in (("baseline", False), ("SAL", True)):
        extra = f", SAL prune_fraction={cfg['prune_fraction']}" if use_sal else ""
        print(f"\n[seed {seed}/{arm}] training{extra}...", flush=True)
        model = ViTForImageClassification.from_pretrained(
            cfg["model"], num_labels=10, ignore_mismatched_sizes=True)
        t0 = time.time()
        trained = train_arm(model, train_b, device, use_sal=use_sal, seed=seed, cfg=cfg)
        print(f"    trained in {time.time() - t0:.0f}s", flush=True)

        # Park the master on CPU: each variant deep-copies from here.
        master = trained.cpu()
        del trained, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results[arm] = run_battery(master, eval_b, device, battery_seed,
                                   f"seed {seed}/{arm}")
        del master
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"seed": seed, "model": cfg["model"], "task": cfg["task"],
            "finetune": "full", "epochs": cfg["epochs"], "lr": cfg["lr"],
            "batch_size": cfg["train_bs"], "n_train": n_train, "n_eval": n_eval,
            "prune_fraction": cfg["prune_fraction"], "battery_seed": battery_seed,
            "baseline": results["baseline"], "SAL": results["SAL"]}


# ------------------------------------------------------------------- reporting
def _mean_std(values):
    """(mean, sample std). std is None for fewer than two values."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def per_seed_table(runs: list) -> str:
    head = (f"{'seed':>6}  {'variant':<14}{'baseline_acc':>14}{'sal_acc':>10}"
            f"{'delta':>10}{'winner':>10}")
    lines = [head, "-" * len(head)]
    for run in runs:
        for v in VARIANTS:
            b = run["baseline"].get(v, {}).get("accuracy")
            s = run["SAL"].get(v, {}).get("accuracy")
            if b is None or s is None:
                lines.append(f"{run['seed']:>6}  {v:<14}{'-':>14}{'-':>10}{'-':>10}"
                             f"{'n/a':>10}")
                continue
            d = s - b
            winner = "SAL" if d > 0 else ("baseline" if d < 0 else "tie")
            lines.append(f"{run['seed']:>6}  {v:<14}{b:>14.4f}{s:>10.4f}{d:>+10.4f}"
                         f"{winner:>10}")
        lines.append("")
    return "\n".join(lines).rstrip()


def aggregate(runs: list) -> dict:
    """Per-variant mean ± std across seeds, plus the unanimity test."""
    agg = {}
    for v in VARIANTS:
        b = [r["baseline"][v]["accuracy"] for r in runs
             if r["baseline"].get(v, {}).get("accuracy") is not None]
        s = [r["SAL"][v]["accuracy"] for r in runs
             if r["SAL"].get(v, {}).get("accuracy") is not None]
        deltas = [r["SAL"][v]["accuracy"] - r["baseline"][v]["accuracy"] for r in runs
                  if r["baseline"].get(v, {}).get("accuracy") is not None
                  and r["SAL"].get(v, {}).get("accuracy") is not None]
        bm, bs = _mean_std(b)
        sm, ss = _mean_std(s)
        dm, ds = _mean_std(deltas)
        sal_wins = sum(1 for d in deltas if d > 0)
        agg[v] = {
            "n_seeds": len(deltas),
            "baseline_mean": bm, "baseline_std": bs,
            "sal_mean": sm, "sal_std": ss,
            "delta_mean": dm, "delta_std": ds,
            "sal_wins": sal_wins, "baseline_wins": sum(1 for d in deltas if d < 0),
            # With n=3 the only defensible bar is unanimity. 2/3 is a coin flip.
            "consistent": bool(deltas) and sal_wins == len(deltas),
        }
    return agg


def aggregate_table(agg: dict) -> str:
    head = (f"{'variant':<14}{'baseline (mean+-std)':>24}{'SAL (mean+-std)':>24}"
            f"{'delta (mean+-std)':>24}{'wins':>7}{'consist?':>10}")
    lines = [head, "-" * len(head)]

    def cell(mean, std, signed=False):
        if mean is None:
            return "-"
        fmt = f"{mean:+.4f}" if signed else f"{mean:.4f}"
        return f"{fmt} +- {std:.4f}" if std is not None else f"{fmt} +- n/a"

    for v, a in agg.items():
        wins = "{}/{}".format(a["sal_wins"], a["n_seeds"])
        lines.append(
            f"{v:<14}{cell(a['baseline_mean'], a['baseline_std']):>24}"
            f"{cell(a['sal_mean'], a['sal_std']):>24}"
            f"{cell(a['delta_mean'], a['delta_std'], signed=True):>24}"
            f"{wins:>7}{('YES' if a['consistent'] else 'NO'):>10}")
    return "\n".join(lines)


def verdict(agg: dict, n_eval: int) -> str:
    compressed = [v for v in VARIANTS if v != "dense"]
    won = [v for v in compressed if agg[v]["consistent"]]
    lost = [v for v in compressed
            if agg[v]["n_seeds"] and agg[v]["baseline_wins"] == agg[v]["n_seeds"]]
    best = max((agg[v]["delta_mean"] or 0) for v in compressed) if compressed else 0

    out = []
    if not any(a["n_seeds"] for a in agg.values()):
        return "verdict: no variant completed on any seed."
    if len(won) == len(compressed):
        out.append("verdict: SAL leads every compressed variant on all seeds — the "
                   "mechanism transfers to vision transformers.")
    elif won:
        out.append(f"verdict: SAL leads {len(won)}/{len(compressed)} compressed "
                   f"variants unanimously ({', '.join(won)}).")
    else:
        out.append("verdict: NO compressed variant favours SAL on every seed. On "
                   "this model and task the mechanism does not transfer.")
    if lost:
        out.append(f"  baseline leads unanimously on: {', '.join(lost)}")

    # The specific claim this run exists to check.
    out.append(f"\n  Largest mean gain on any compressed variant: {best * 100:+.2f}pp.")
    if best < 0.05:
        out.append("  The original research reported +20pp on ViT. Nothing here is "
                   "within an order of magnitude of that, so it does NOT reproduce.")
    else:
        out.append("  Compare against the +20pp the original research reported on ViT.")
    if n_eval:
        out.append(f"  noise floor: {n_eval} eval images, so one image is "
                   f"{1.0 / n_eval:.3%}.")
    out.append("  Three seeds. 'consist?' is unanimity, not significance.")
    return "\n".join(out)


@app.local_entrypoint()
def main():
    cfg = CONFIG
    print(f"ViT validation: {cfg['model']} / {cfg['task']}, seeds {SEEDS}, "
          f"{cfg['epochs']} epochs, full fine-tuning"
          + ("   [SMOKE MODE — not a result]" if cfg["smoke"] else ""))

    runs, failures = [], {}
    remote = run_seed.map(SEEDS, kwargs={"cfg": cfg}, return_exceptions=True)
    for seed, res in zip(SEEDS, remote):
        if isinstance(res, Exception):
            print(f"seed {seed} failed: {res}")
            failures[str(seed)] = str(res)
            continue
        runs.append(res)

    if not runs:
        raise SystemExit("every seed failed — nothing to report")

    runs.sort(key=lambda r: SEEDS.index(r["seed"]))
    n_eval = runs[0]["n_eval"]

    print("\n########## PER-SEED RESULTS ##########")
    print(per_seed_table(runs))

    agg = aggregate(runs)
    print(f"\n########## AGGREGATED OVER {len(runs)} SEED(S) ##########")
    print(aggregate_table(agg))
    print()
    print(verdict(agg, n_eval))

    payload = {
        "config": {**cfg, "finetune": "full", "seeds": SEEDS, "n_eval": n_eval,
                   "head_selection": "random (shipped default)",
                   "int8_backend": "torch.ao dynamic (CPU)"},
        "seeds_completed": [r["seed"] for r in runs],
        "failures": failures,
        "runs": {str(r["seed"]): r for r in runs},
        "aggregate": agg,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
