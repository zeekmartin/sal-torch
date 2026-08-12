"""Multi-seed validation — does the v0.4.0 compression sweep survive more seeds?

Every SAL-vs-standard number published so far is a **single seed**, and this
project already knows what that is worth: two GPT-2 Medium baselines trained on
identical data differ by 0.58pp, and the v0.5.0 pipeline run reversed the v0.4.0
result on the same model and task. ROADMAP.md has been calling for more seeds
since v0.4.0 shipped. This is that run, and v0.5.0 does not get tagged until it
finishes.

Protocol
--------
The v0.4.0 ``gpt2_full`` tier, repeated five times with a different seed each
time and scored on the **full SST-2 validation split** (872 examples) instead of
a 512-example subset.

Per seed, both arms train from the same pretrained checkpoint:

* **baseline** — GPT-2 Medium, full fine-tuning, 3 epochs, lr=2e-5, batch 8.
* **SAL** — identical, plus ``HeadMasker`` at ``prune_fraction=0.33``.

Full fine-tuning, never LoRA. SAL's mechanism is the model reorganizing around
silenced heads; adapters freeze the weights that would do the reorganizing, and
the measured cost of getting that wrong is about three accuracy points.

Then seven compression variants per arm::

    dense, int8, int4, prune33, prune50, prune33+int8, prune33+int4

* ``int8`` is ``torch.ao`` dynamic quantization (CPU-only, so those variants are
  scored on CPU). **This differs from the v0.4.0 run**, which used bitsandbytes
  LLM.int8() on CUDA — read the int8 rows as a different measurement, not a
  replication. The int4 rows are bitsandbytes NF4 exactly as before.
* ``prune33`` / ``prune50`` mask randomly chosen heads — ``random`` is the
  shipped default selection strategy, and the v0.5.0 grid measured it as the
  best of the three for *both* arms.
* Combined variants quantize first, then install the head mask, so the pruning
  hooks land on the quantized modules rather than being deep-copied through them.

The eval-time head choice uses a seed **derived separately** from the training
seed (``10000 + seed``), so the heads SAL trained against are not the heads the
battery removes. Both arms of a seed lose the *same* heads, which is what makes
the row a comparison.

What counts as a result
-----------------------
Two tables. Per-seed rows, then mean ± std across seeds with a ``significant?``
column that is YES only when SAL wins **4 or more of the 5 seeds** for that
variant. A variant that splits 3/2 is noise, whichever way it leans, and gets
NO. The 4/5 rule is a sign test, not a t-test: it asks whether the direction is
consistent, which is the question a single seed cannot answer.

If SAL does not win, this script prints that it did not. The README and ROADMAP
get whatever this produces.

Usage::

    modal run scripts/modal_multiseed_validation.py            # the real run
    SAL_SMOKE=1 modal run scripts/modal_multiseed_validation.py  # plumbing check

Environment overrides: ``SAL_SEEDS`` (comma-separated), ``SAL_N_TRAIN``,
``SAL_N_EVAL`` (0 = full split), ``SAL_EPOCHS``. Results are written to
``scripts/multiseed_results.json``.
"""
from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("sal-torch-multiseed")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "numpy", "accelerate>=1.1.0",
                 "bitsandbytes")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

VARIANTS = ["dense", "int8", "int4", "prune33", "prune50", "prune33+int8", "prune33+int4"]

# The protocol. Everything that varies between a smoke run and the real one lives
# here and is passed to the remote function **as an argument**, never read from
# the environment inside the container: `modal run` does not forward the local
# environment, so a container reading `os.environ` would quietly ignore every
# override and run the full protocol while the local side reported a smoke test.
BASE_CONFIG = {
    "model": "gpt2-medium",
    "task": "sst2",
    "n_train": 1024,
    "n_eval": 0,            # 0 = the full SST-2 validation split (872 examples)
    "epochs": 3,
    "lr": 2e-5,
    "train_bs": 8,
    "grad_accum": 1,
    "eval_bs": 16,
    "max_len": 96,
    "prune_fraction": 0.33,
    # Keeps the eval-time random head choice independent of the heads SAL was
    # trained against, while staying identical across the two arms of a seed.
    "battery_seed_offset": 10_000,
}


def _build_config():
    """(seeds, config) for this invocation, from the optional env overrides."""
    smoke = os.environ.get("SAL_SMOKE", "").strip() not in ("", "0", "false")
    seeds = [int(s) for s in
             os.environ.get("SAL_SEEDS", "42,123,456,789,1337").split(",") if s.strip()]
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
# A smoke run must never be mistaken for the published result, so it does not get
# to write the file the README cites.
RESULTS_PATH = ("scripts/multiseed_smoke.json" if CONFIG["smoke"]
                else "scripts/multiseed_results.json")


# ------------------------------------------------------------------ task loading
def load_task(n_train: int, n_eval: int, seed: int):
    """(train, eval) as lists of {prompt, choices, gold}. Eval is the full split."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2")

    def conv(r):
        return {"prompt": f'Review: "{r["sentence"].strip()}"\nSentiment:',
                "choices": [" negative", " positive"], "gold": int(r["label"])}

    # Only the training subset is reshuffled per seed. The eval split is held
    # fixed and complete, so every seed is scored on exactly the same examples.
    train = [conv(r) for r in ds["train"].shuffle(seed=seed).select(range(n_train))]
    ev = ds["validation"]
    if n_eval:
        ev = ev.select(range(min(n_eval, len(ev))))
    return train, [conv(r) for r in ev]


# --------------------------------------------------------- multiple-choice scoring
def _encode_pair(tok, prompt: str, continuation: str, max_len: int):
    """Token ids for prompt+continuation, plus how many trailing tokens are the
    continuation (after truncation from the left)."""
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    c = tok(continuation, add_special_tokens=False)["input_ids"]
    ids = (p + c)[-max_len:]
    n_cont = min(len(c), len(ids) - 1)     # need >= 1 context token to predict from
    return ids, max(n_cont, 1)


def _score_pairs(model, tok, pairs, max_len: int, device):
    """Length-normalized log P(continuation | prompt) for each (prompt, cont) pair."""
    import torch

    encoded = [_encode_pair(tok, p, c, max_len) for p, c in pairs]
    width = max(len(ids) for ids, _ in encoded)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, (ids, _) in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1

    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
        logprobs = torch.log_softmax(out.logits.float(), dim=-1)

    scores = []
    for i, (ids, n_cont) in enumerate(encoded):
        n = len(ids)
        target = input_ids[i, n - n_cont:n].to(logprobs.device)
        # Token at position t is predicted by the logits at position t-1.
        window = logprobs[i, n - n_cont - 1:n - 1, :]
        lp = window.gather(-1, target.unsqueeze(-1)).sum()
        scores.append(float(lp) / n_cont)
    return scores


def evaluate_accuracy(model, tok, examples, max_len: int, batch_size: int, device):
    """Multiple-choice accuracy by length-normalized log-likelihood."""
    model.eval()
    flat = [(i, j, ex["prompt"], ch)
            for i, ex in enumerate(examples) for j, ch in enumerate(ex["choices"])]
    scored = {}
    for start in range(0, len(flat), batch_size):
        chunk = flat[start:start + batch_size]
        vals = _score_pairs(model, tok, [(p, c) for _, _, p, c in chunk], max_len, device)
        for (i, j, _, _), v in zip(chunk, vals):
            scored.setdefault(i, {})[j] = v
    correct = 0
    for i, ex in enumerate(examples):
        per_choice = scored.get(i, {})
        if not per_choice:
            continue
        pred = max(per_choice, key=per_choice.get)
        correct += int(pred == ex["gold"])
    return correct / max(len(examples), 1)


# --------------------------------------------------------------------- training
def _train_batch(tok, examples, idxs, max_len: int):
    """Causal-LM batch over prompt+gold continuation, loss masked to the continuation."""
    import torch

    rows = []
    for j in idxs:
        ex = examples[j]
        ids, n_cont = _encode_pair(tok, ex["prompt"], ex["choices"][ex["gold"]], max_len)
        labels = [-100] * len(ids)
        for t in range(len(ids) - n_cont, len(ids)):
            labels[t] = ids[t]
        rows.append((ids, labels))

    width = max(len(ids) for ids, _ in rows)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (ids, labs) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)
        attn[i, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def train_arm(model, tok, examples, device, use_sal: bool, seed: int, cfg: dict):
    """Fully fine-tune one arm, optionally with SAL head masking.

    Every parameter is trainable — no adapters anywhere. The masker is installed
    after ``.to(device)`` so its masks are allocated where the model lives, and
    removed before returning so the compression battery sees a plain model whose
    weights already carry the adaptation.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    from sal.config import SALConfig
    from sal.masker import HeadMasker

    epochs, accum, bs = cfg["epochs"], cfg["grad_accum"], cfg["train_bs"]

    torch.manual_seed(seed)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    masker = None
    if use_sal:
        config = SALConfig.auto(model, prune_fraction=cfg["prune_fraction"])
        masker = HeadMasker(model, config, seed=seed)
        masker.install()

    order = np.random.RandomState(seed).permutation(len(examples))
    batches = [order[i:i + bs] for i in range(0, len(examples), bs)]
    total_steps = max(1, (len(batches) * epochs) // accum)

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"    full FT: {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M trainable "
          f"({n_trainable / max(n_total, 1):.1%}), lr={cfg['lr']:g}, {total_steps} steps",
          flush=True)

    opt = AdamW(params, lr=cfg["lr"])
    model.train()

    step, micro = 0, 0
    for _ in range(epochs):
        for idxs in batches:
            if masker is not None:
                masker.step(step, total_steps)
            batch = {k: v.to(device) for k, v in
                     _train_batch(tok, examples, idxs, cfg["max_len"]).items()}
            loss = model(**batch).loss / accum
            loss.backward()
            micro += 1
            if micro % accum == 0:
                # Silencing a third of the heads makes gradients spikier than
                # ordinary fine-tuning; unclipped, this run loses ~13 points.
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
    """Materialize one compression variant. Returns (model, backend, device, masker).

    Quantization runs before masking so the pruning hooks land on the quantized
    modules — deep-copying a model that already carries hooks is a trap.
    """
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
    elif "int4" in variant:
        model = quantize(model, method="int4", backend="bitsandbytes", inplace=True)
        backend, dev = "bitsandbytes-nf4", "cuda"
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


def run_battery(master, tok, examples, device, battery_seed: int, label: str,
                cfg: dict) -> dict:
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
            acc = evaluate_accuracy(model, tok, examples, cfg["max_len"],
                                    cfg["eval_bs"], dev)
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
@app.function(image=image, gpu="T4", cpu=16.0, memory=32768, timeout=3600)
def run_seed(seed: int, cfg: dict) -> dict:
    """Train both arms at this seed and score the full battery on each."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 8)     # the int8 variants run here
    print(f"=== seed {seed}: {cfg['model']} / {cfg['task']}, full fine-tuning "
          f"on {device} ===", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ex, eval_ex = load_task(cfg["n_train"], cfg["n_eval"], seed)
    print(f"    {len(train_ex)} train / {len(eval_ex)} eval examples "
          f"({'subset' if cfg['n_eval'] else 'full validation split'}), "
          f"{cfg['epochs']} epochs, lr={cfg['lr']:g}, batch {cfg['train_bs']}",
          flush=True)

    battery_seed = cfg["battery_seed_offset"] + seed
    results = {}
    for arm, use_sal in (("baseline", False), ("SAL", True)):
        extra = f", SAL prune_fraction={cfg['prune_fraction']}" if use_sal else ""
        print(f"\n[seed {seed}/{arm}] training{extra}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
        t0 = time.time()
        trained = train_arm(model, tok, train_ex, device, use_sal=use_sal, seed=seed,
                            cfg=cfg)
        print(f"    trained in {time.time() - t0:.0f}s", flush=True)

        # Park the master on CPU: each variant deep-copies from here.
        master = trained.cpu()
        del trained, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results[arm] = run_battery(master, tok, eval_ex, device, battery_seed,
                                   f"seed {seed}/{arm}", cfg)
        del master
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"seed": seed, "model": cfg["model"], "task": cfg["task"],
            "finetune": "full", "epochs": cfg["epochs"], "lr": cfg["lr"],
            "batch_size": cfg["train_bs"], "n_train": len(train_ex),
            "n_eval": len(eval_ex), "prune_fraction": cfg["prune_fraction"],
            "battery_seed": battery_seed,
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
    """Per-variant mean ± std across seeds, plus the 4-of-5 sign test."""
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
            "ties": sum(1 for d in deltas if d == 0),
            # A sign test, not a t-test: does the direction hold across seeds?
            "significant": bool(deltas) and sal_wins >= 4,
        }
    return agg


def aggregate_table(agg: dict) -> str:
    head = (f"{'variant':<14}{'baseline (mean+-std)':>24}{'SAL (mean+-std)':>24}"
            f"{'delta (mean+-std)':>24}{'wins':>7}{'signif?':>9}")
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
            f"{wins:>7}{('YES' if a['significant'] else 'NO'):>9}")
    return "\n".join(lines)


def verdict(agg: dict, n_eval: int) -> str:
    wins = [v for v, a in agg.items() if a["significant"]]
    losses = [v for v, a in agg.items()
              if a["n_seeds"] and a["baseline_wins"] >= 4]
    compressed = [v for v in VARIANTS if v != "dense"]
    won = [v for v in wins if v != "dense"]

    out = []
    if not any(a["n_seeds"] for a in agg.values()):
        return "verdict: no variant completed on any seed."
    if len(won) == len(compressed):
        out.append("verdict: SAL wins every compressed variant on 4+ of 5 seeds — "
                   "the v0.4.0 sweep replicates.")
    elif won:
        out.append(f"verdict: SAL wins {len(won)}/{len(compressed)} compressed "
                   f"variants on 4+ of 5 seeds ({', '.join(won)}). The rest are "
                   f"not consistent across seeds.")
    else:
        out.append("verdict: NO compressed variant favours SAL on 4+ of 5 seeds. "
                   "The single-seed v0.4.0 sweep does not replicate.")
    if losses:
        out.append(f"  baseline wins 4+ of 5 seeds on: {', '.join(losses)}")
    if n_eval:
        out.append(f"  noise floor: {n_eval} eval examples, so one example is "
                   f"{1.0 / n_eval:.3%}.")
    out.append("  'significant?' is a sign test (SAL ahead on 4+ of 5 seeds), not a "
               "t-test. It asks whether the direction is consistent.")
    return "\n".join(out)


@app.local_entrypoint()
def main():
    cfg = CONFIG
    print(f"multi-seed validation: {cfg['model']} / {cfg['task']}, seeds {SEEDS}, "
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

    print("\n########## TABLE 1 — PER-SEED RESULTS ##########")
    print(per_seed_table(runs))

    agg = aggregate(runs)
    print(f"\n########## TABLE 2 — AGGREGATED OVER {len(runs)} SEED(S) ##########")
    print(aggregate_table(agg))
    print()
    print(verdict(agg, n_eval))

    payload = {
        "config": {**cfg, "finetune": "full", "seeds": SEEDS, "n_eval": n_eval,
                   "eval_split": (f"sst2 validation[:{cfg['n_eval']}]"
                                  if cfg["n_eval"] else "sst2 validation (full)"),
                   "head_selection": "random (shipped default)",
                   "int8_backend": "torch.ao dynamic (CPU)",
                   "int4_backend": "bitsandbytes NF4 (CUDA)"},
        "seeds_completed": [r["seed"] for r in runs],
        "failures": failures,
        "runs": {str(r["seed"]): r for r in runs},
        "aggregate": agg,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
