"""SAL on I-JEPA: does a self-supervised encoder survive head pruning better?

The v0.5.0 evidence for SAL is supervised and textual — GPT-2 on SST-2, scored
by accuracy. This run asks whether the effect carries to a self-supervised
*vision* encoder, where there is no head and no accuracy, only representations.

The protocol, on ImageNet-100:

  1. scan the pretrained I-JEPA encoder (FI + absorption map)
  2. train **two** arms with the *same* objective, optimizer, data and step
     count — one with SAL head masking, one without. The no-SAL arm is trained,
     not pristine; otherwise the comparison is "trained vs untrained" wearing a
     compression costume.
  3. prune each arm at 33% and 50%, under random and magnitude head selection
  4. score every arm on linear probe, kNN, CKA against **its own** unpruned
     model, parameter count and latency

**Model.** ``facebook/ijepa_vith14_1k`` — 632M parameters, 32 layers x 16 heads.
Meta never released an I-JEPA ViT-B/16; H/14 and g/16 are the whole catalogue,
and H/14 is the smaller.

Currently set to **A10G (24GB)**, which is a deliberate experiment rather than a
settled sizing. The weights are small in bf16, but that is not what dominates:
the optimizer runs in fp32 (autocast casts activations, not parameters), so
AdamW holds roughly 2.5GB of params + 2.5GB of grads + 5GB of moments, and the
run keeps *both* arms resident at once so the SAL and control models are scored
against the same probe. Add eager attention's stored attention matrices on top.
If the smoke test OOMs, the levers in order are ``gradient_checkpointing_enable()``,
``BATCH_SIZE = 1``, freeing each arm between phases, and then A100-40GB.

**Objective.** I-JEPA-*shaped*, not I-JEPA. Real I-JEPA has a separate predictor
network and an EMA target encoder; reproducing that needs the pretraining
apparatus. What runs here keeps the part that matters — predict the
representation of the whole image from a partially visible one — and drops the
predictor and the EMA: the target is the clean image through the unperturbed
model under no_grad, the prediction is a patch-masked image through the
(SAL-perturbed) model, and the loss is MSE between them.

The patch masking is not decoration. The brief's version used the clean image on
both sides, and with head masking off that loss is *identically zero* — the
control arm would run its optimizer on a constant and learn nothing, while
looking in every log like it had trained. Masking input patches gives both arms
a real gradient, so "SAL vs standard" compares two trained models and SAL head
masking is the only variable between them.

**What would falsify the claim.** If SAL and standard arms land within noise of
each other on kNN and CKA at both ratios, SAL does not transfer to this setting.
A single seed cannot distinguish a small real effect from seed noise — the
v0.5.0 five-seed run found int4 gains evaporating that way — so read a gap under
~1pp here as "not shown", and run ``--seeds`` before quoting anything.

Usage::

    modal run scripts/modal_jepa_sal.py --smoke      # ~15 min, validates the path
    modal run scripts/modal_jepa_sal.py              # the real run, ~4h, A100

Results are written to ``data/results/jepa_sal_benchmark.json``.
"""
from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("sal-torch-jepa-sal")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers", "datasets", "numpy",
                 "accelerate>=1.1.0", "pillow")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

# Cache the 2.5GB checkpoint and the 8.4GB dataset across runs.
cache = modal.Volume.from_name("sal-jepa-cache", create_if_missing=True)

MODEL_ID = "facebook/ijepa_vith14_1k"
DATASET_ID = "clane9/imagenet-100"

EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-5
MASK_RATIO = 0.30        # fraction of attention heads SAL prunes while training
PATCH_MASK_RATIO = 0.40  # fraction of input patches hidden from the prediction pass
PRUNE_RATIOS = (0.33, 0.50)
SELECTION_METHODS = ("random", "magnitude")

N_TRAIN = 8192         # SAL training images
N_PROBE_TRAIN = 5000   # linear probe / kNN support set
N_PROBE_VAL = 2500     # linear probe / kNN query set
N_FI_PROBE = 256       # images for the structural scans

RESULTS_PATH = "data/results/jepa_sal_benchmark.json"


# --------------------------------------------------------------------- data
def _build_loaders(smoke: bool, image_size: int, seed: int):
    """ImageNet-100 as (pixel_values, label) tensor batches."""
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor

    n_train = 100 if smoke else N_TRAIN
    n_ptr = 100 if smoke else N_PROBE_TRAIN
    n_pva = 50 if smoke else N_PROBE_VAL
    n_fi = 32 if smoke else N_FI_PROBE

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID)
    train_split = ds["train"].shuffle(seed=seed)
    val_split = ds["validation"].shuffle(seed=seed)

    def encode(split, n):
        subset = split.select(range(min(n, len(split))))
        pixels, labels = [], []
        for i in range(0, len(subset), 64):
            chunk = subset[i:i + 64]
            imgs = [im.convert("RGB") for im in chunk["image"]]
            pixels.append(proc(imgs, return_tensors="pt")["pixel_values"])
            labels.append(torch.tensor(chunk["label"]))
        return torch.cat(pixels), torch.cat(labels)

    xtr, ytr = encode(train_split, n_train)
    xpt, ypt = encode(train_split.select(range(n_train, len(train_split))), n_ptr)
    xpv, ypv = encode(val_split, n_pva)

    num_classes = int(max(ytr.max(), ypt.max(), ypv.max())) + 1
    loaders = {
        # shuffle=False everywhere: CKA pairs features example by example.
        "train": DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH_SIZE, shuffle=True),
        "probe_train": DataLoader(TensorDataset(xpt, ypt), batch_size=32),
        "probe_val": DataLoader(TensorDataset(xpv, ypv), batch_size=32),
        "cka": DataLoader(TensorDataset(xpv[:n_fi]), batch_size=32),
    }
    fi_batches = [{"pixel_values": xpv[i:i + 16]} for i in range(0, n_fi, 16)]
    return loaders, fi_batches, num_classes


# --------------------------------------------------------------- the objective
def _mask_patches(pixels, patch_size: int, ratio: float, generator=None):
    """Zero a random ``ratio`` of non-overlapping patches, per image.

    Each image in the batch gets its own mask — one shared pattern would let the
    model memorize which positions are always hidden.
    """
    import torch

    b, c, h, w = pixels.shape
    gh, gw = h // patch_size, w // patch_size
    n = gh * gw
    keep = torch.rand(b, n, generator=generator, device=pixels.device) >= ratio
    mask = keep.view(b, 1, gh, 1, gw, 1).to(pixels.dtype)
    patched = pixels.view(b, c, gh, patch_size, gw, patch_size)
    return (patched * mask).view(b, c, h, w)


def _make_step(patch_size: int, use_amp: bool, patch_mask_ratio: float = PATCH_MASK_RATIO):
    """I-JEPA-shaped step: predict the whole-image representation from a
    partially visible image, through a SAL-perturbed model.

    The target pass uses ``mask_module.unmasked()`` rather than ``deactivate()``
    — the latter refills the masks with ones and would throw away the pruned set
    the schedule has accumulated, silently undoing SAL.
    """
    import torch

    def step(model, batch, optimizer, mask_module):
        pixels = batch[0]
        visible = _mask_patches(pixels, patch_size, patch_mask_ratio)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad(), mask_module.unmasked():
                target = model(pixel_values=pixels).last_hidden_state
            predicted = model(pixel_values=visible).last_hidden_state
            loss = torch.nn.functional.mse_loss(predicted, target.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    return step


def _train_standard(model, loader, epochs, lr, patch_size, use_amp):
    """The control arm: identical objective and step count, no head masking.

    Deliberately not SALTrainer with a near-zero prune_fraction — the schedule
    would still fire and prune at least one head (num_heads_to_prune has a floor
    of 1). The control has to be exactly zero masking, so it gets its own short
    loop rather than a configuration that is merely *nearly* off.
    """
    import torch

    class _NoMask:
        """Stands in for the masker: every control is a no-op."""
        def apply_mask(self): pass
        def remove_mask(self): pass
        def unmasked(self):
            import contextlib
            return contextlib.nullcontext()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step = _make_step(patch_size, use_amp)
    model.train()
    losses = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for batch in loader:
            batch = [b.cuda() if torch.is_tensor(b) else b for b in batch]
            total += step(model, batch, opt, _NoMask()); n += 1
        losses.append(total / max(n, 1))
    return {"losses": losses}


def _train_sal(model, loader, epochs, lr, mask_ratio, seed, patch_size, use_amp):
    import torch
    from sal import SALConfig
    from sal.trainer import SALTrainer

    cfg = SALConfig.auto(model, prune_fraction=mask_ratio)
    trainer = SALTrainer(model, cfg, torch.optim.AdamW(model.parameters(), lr=lr),
                         loader, seed=seed,
                         train_step=_make_step(patch_size, use_amp))
    return trainer.train(num_epochs=epochs)


# ------------------------------------------------------------------- pruning
def _prune(model, ratio, method, seed=0):
    """A copy of ``model`` with ``ratio`` of heads masked off (not sliced).

    Masking keeps the parameter count identical across arms, so the comparison
    isolates *function* from *size*. sal.slice_heads() is what turns a chosen
    ratio into an actual smaller model.
    """
    import copy

    import torch

    from sal import SALConfig, arch_support
    from sal.masker import HeadMasker

    pruned = copy.deepcopy(model)
    info = arch_support.detect_architecture(pruned)
    cfg = SALConfig.auto(pruned, prune_fraction=ratio, prune_start_ratio=0.0)
    masker = HeadMasker(pruned, cfg, seed=seed)
    masker.install()

    if method == "random":
        masker.activate()
    else:                                        # magnitude: weakest heads first
        masker.apply_mask()
        n_drop = int(round(ratio * info.num_heads))
        projs = arch_support.get_output_projections(pruned, info.attention_pattern)
        for layer_idx, proj in enumerate(projs):
            w = proj.weight.detach()
            per_head = w.view(w.shape[0], info.num_heads, -1).norm(dim=(0, 2))
            for head in torch.argsort(per_head)[:n_drop].tolist():
                masker._masks[layer_idx][head] = 0.0

    pruned._sal_masker = masker                  # keep the hooks alive
    return pruned


# ----------------------------------------------------------------- the run
@app.function(image=image, gpu="A10G", timeout=4 * 60 * 60,
              volumes={"/cache": cache})
def run(smoke: bool = False, seed: int = 42) -> dict:
    import torch
    from transformers import AutoModel

    from sal import (FIScanner, PlasticityScanner, cka_similarity, count_params,
                     knn_accuracy, linear_probe, measure_latency)

    os.environ.setdefault("HF_HOME", "/cache/hf")
    torch.manual_seed(seed)
    started = time.time()
    epochs = 1 if smoke else EPOCHS
    use_amp = True

    print(f"=== loading {MODEL_ID} ===", flush=True)
    # eager attention: the plasticity scanner reads attention weights, and SDPA
    # returns none — routing entropy comes back NaN and hub detection reports
    # "no hubs", which reads exactly like a finding.
    base = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    image_size = base.config.image_size
    patch_size = base.config.patch_size
    print(f"    {count_params(base)/1e6:.0f}M params, "
          f"{base.config.num_hidden_layers}x{base.config.num_attention_heads} heads",
          flush=True)

    loaders, fi_batches, num_classes = _build_loaders(smoke, image_size, seed)
    print(f"    {num_classes} classes, {len(loaders['train'].dataset)} train images",
          flush=True)

    fi_batches = [{k: v.cuda() for k, v in b.items()} for b in fi_batches]
    base = base.cuda()

    # 1 ------------------------------------------------------ pre-SAL scan
    print("=== pre-SAL structural scan ===", flush=True)
    scan_before = FIScanner(base, fi_batches, num_samples=len(fi_batches) * 16).scan()
    plasticity = PlasticityScanner(base, fi_batches,
                                   num_samples=len(fi_batches) * 16).scan()
    print(f"    {scan_before.summary}", flush=True)

    import copy
    sal_model = copy.deepcopy(base)
    std_model = copy.deepcopy(base)
    del base
    torch.cuda.empty_cache()

    # 2 --------------------------------------------------------- both arms
    print(f"=== SAL arm: {epochs} epochs, mask_ratio={MASK_RATIO} ===", flush=True)
    sal_hist = _train_sal(sal_model, loaders["train"], epochs, LR, MASK_RATIO,
                          seed, patch_size, use_amp)
    print(f"    losses: {[round(x, 6) for x in sal_hist['losses']]}", flush=True)
    print(f"    masker: {sal_hist['masker_stats']}", flush=True)

    print(f"=== standard arm: {epochs} epochs, no masking ===", flush=True)
    std_hist = _train_standard(std_model, loaders["train"], epochs, LR, patch_size,
                               use_amp)
    print(f"    losses: {[round(x, 6) for x in std_hist['losses']]}", flush=True)

    # 3 ----------------------------------------------------- post-SAL scan
    print("=== post-SAL structural scan ===", flush=True)
    scan_after = FIScanner(sal_model, fi_batches,
                           num_samples=len(fi_batches) * 16).scan()
    print(f"    FI {scan_before.fi_score:.4f} -> {scan_after.fi_score:.4f}", flush=True)

    # 4 ------------------------------------------------------- benchmark
    print("=== compression benchmark ===", flush=True)
    image_shape = (1, 3, image_size, image_size)
    results = {}
    for arm, model in (("sal", sal_model), ("standard", std_model)):
        # Each arm's clean scores are its own reference: "how much did pruning
        # cost *this* model", not "which model is better in absolute terms".
        results[f"{arm}_clean"] = {
            "linear_probe": linear_probe(model, loaders["probe_train"],
                                         loaders["probe_val"], num_classes=num_classes),
            "knn_accuracy": knn_accuracy(model, loaders["probe_train"],
                                         loaders["probe_val"], k=20),
            "cka_similarity": 1.0,
            "params": count_params(model),
            "latency_gpu_ms": measure_latency(model, image_shape, "cuda"),
        }
        print(f"    {arm}_clean: {results[f'{arm}_clean']}", flush=True)

        for ratio in PRUNE_RATIOS:
            for method in SELECTION_METHODS:
                name = f"{arm}_{method}_{int(ratio * 100)}"
                pruned = _prune(model, ratio, method, seed=seed)
                results[name] = {
                    "linear_probe": linear_probe(pruned, loaders["probe_train"],
                                                 loaders["probe_val"],
                                                 num_classes=num_classes),
                    "knn_accuracy": knn_accuracy(pruned, loaders["probe_train"],
                                                 loaders["probe_val"], k=20),
                    "cka_similarity": cka_similarity(model, pruned, loaders["cka"]),
                    "params": count_params(pruned),
                    "latency_gpu_ms": measure_latency(pruned, image_shape, "cuda"),
                }
                print(f"    {name}: {results[name]}", flush=True)
                del pruned
                torch.cuda.empty_cache()

    return {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "smoke": smoke,
        "seed": seed,
        "epochs": epochs,
        "mask_ratio": MASK_RATIO,
        "patch_mask_ratio": PATCH_MASK_RATIO,
        "prune_ratios": list(PRUNE_RATIOS),
        "objective": "I-JEPA-shaped: predict the clean-image representation from "
                     "a patch-masked image. No predictor network, no EMA target "
                     "encoder — NOT I-JEPA pretraining.",
        "num_classes": num_classes,
        "fi_before": scan_before.fi_score,
        "fi_after_sal": scan_after.fi_score,
        "layer_classification_before": {str(k): v.value
                                        for k, v in scan_before.layer_map.items()},
        "absorption_map_before": {str(k): v for k, v in plasticity.absorption_map.items()},
        "sal_losses": sal_hist["losses"],
        "standard_losses": std_hist["losses"],
        "masker_stats": sal_hist["masker_stats"],
        "results": results,
        "wall_clock_s": round(time.time() - started, 1),
    }


# ------------------------------------------------------------------ reporting
def _table(results: dict) -> str:
    """SAL vs standard, one row per prune setting, delta in percentage points."""
    lines = [f"{'setting':<16}{'metric':<16}{'standard':>10}{'SAL':>10}{'delta':>10}",
             "-" * 62]
    settings = sorted({k.split("_", 1)[1] for k in results})
    for setting in settings:
        for metric in ("linear_probe", "knn_accuracy", "cka_similarity"):
            std = results.get(f"standard_{setting}", {}).get(metric)
            sal = results.get(f"sal_{setting}", {}).get(metric)
            if std is None or sal is None:
                continue
            lines.append(f"{setting:<16}{metric:<16}{std:>10.4f}{sal:>10.4f}"
                         f"{(sal - std) * 100:>+9.2f}pp")
    lines.append("")
    lines.append("CKA is against each arm's own unpruned model. A gap under ~1pp on "
                 "a single seed is not a result.")
    return "\n".join(lines)


@app.local_entrypoint()
def main(smoke: bool = False, seed: int = 42):
    payload = run.remote(smoke=smoke, seed=seed)

    print()
    print(_table(payload["results"]))
    print()
    print(f"FI: {payload['fi_before']:.4f} -> {payload['fi_after_sal']:.4f} (SAL arm)")
    print(f"wall clock: {payload['wall_clock_s'] / 60:.1f} min")

    out = RESULTS_PATH.replace(".json", "_smoke.json") if smoke else RESULTS_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")
