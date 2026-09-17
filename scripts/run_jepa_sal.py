#!/usr/bin/env python3
"""SAL compression benchmark on I-JEPA ViT-H/14.

Runs on any machine with a GPU. No cloud SDK required — this is the experiment
from ``scripts/modal_jepa_sal.py`` with Modal taken out, so it works over SSH on
RunPod, Lambda, vast.ai, or a box under a desk.

Usage::

    # Quick validation (target: under 5 minutes, excluding the first download)
    python scripts/run_jepa_sal.py --smoke

    # Full benchmark
    python scripts/run_jepa_sal.py --epochs 5 --output results/

    # Resume if interrupted
    python scripts/run_jepa_sal.py --resume results/checkpoint.pt

Requirements::

    pip install -e ".[reports]" datasets torchvision safetensors transformers

(No ``timm``: the I-JEPA checkpoint is on the Hub as an ``IJepaModel``, so
transformers loads it directly. torchvision is needed by the HF image
processor.)

READ THIS BEFORE QUOTING ANY NUMBER
-----------------------------------

**There is no I-JEPA ViT-B/16.** Meta released I-JEPA at ViT-H/14 and ViT-g/16
only. This runs ``facebook/ijepa_vith14_1k`` — 632M parameters, 32 layers x 16
heads.

**The training objective is I-JEPA-shaped, not I-JEPA.** No predictor network,
no EMA target encoder. It predicts the clean-image representation from a
patch-masked image. Using the clean image on *both* sides — the obvious
simplification — makes the loss identically zero whenever head masking is off,
so an unmasked control would train on a constant while logging as though it
were learning.

**By default this table compares head *selection*, not SAL training.** One model
is SAL-trained, then pruned two ways: ``random`` (SAL's own, validated default)
and ``magnitude`` (the standard post-hoc baseline). That is a real question, but
it is not "does SAL training help" — both rows come from the same trained
weights. Pass ``--control`` to also train a no-SAL arm on an identical objective
and step count, which is the comparison that answers the headline question. The
summary prints which one you actually ran.

**A smoke run does not reach the full prune fraction.** The schedule ramps the
pruned-head count across a window and only hits the target once training passes
``prune_end_ratio``. Three optimizer steps never get there, so `--smoke` reports
fewer pruned heads than ``--mask-ratio`` asks for. That is the schedule working.

**The FI column is not comparable across pruned rows.** Measured on the real
ViT-H/14: FI 0.0060 dense, 0.0000 once a third of the heads are masked. That is
an artefact, not a robustness gain. A masked head emits an identically-zero
signature, which correlates with nothing and so carries *no* edges; the graph's
fixed edge budget then redistributes onto the surviving heads, and a denser
subgraph over fewer nodes is more triangulated by construction. Compare FI
between a dense model and a `slice_heads()`-shrunk one, where the node count
matches reality -- never between dense and masked.

**The latency columns show no speedup, and should not.** Pruning here *masks*
heads, so every variant keeps all 631M parameters and does all the same
arithmetic. Measured: 28.8ms dense against 32.0ms pruned -- the pruned rows are
*slower*, by the cost of the masking hooks. Real latency wins come from
`sal.slice_heads()`, which removes the weights. The latency columns are here to
confirm the size is unchanged, not to advertise a gain.

**One seed settles nothing.** The v0.5.0 five-seed run watched an apparent int4
gain evaporate. Treat a sub-1pp gap as "not shown".
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "facebook/ijepa_vith14_1k"
DATASET_ID = "clane9/imagenet-100"

LR = 1e-5
PATCH_MASK_RATIO = 0.40   # fraction of input patches hidden from the prediction
PRUNE_RATIOS = (0.33, 0.50)
SELECTION_METHODS = ("random", "magnitude")

# Full-run sizes; --smoke overrides all of them.
N_TRAIN = 8192
N_PROBE_TRAIN = 5000
N_PROBE_VAL = 2500
N_FI_PROBE = 256
LATENCY_RUNS = 50

DEFAULT_EPOCHS = 5
SMOKE = dict(n_train=100, n_probe_train=100, n_probe_val=50, n_fi=32,
             latency_runs=5, epochs=1)

# Batch size by VRAM. ViT-H/14 trains in fp32 under AdamW even with autocast on
# (autocast casts activations, not parameters), so the optimizer alone wants
# ~10GB before a single activation is stored.
_VRAM_BATCH = [(70, 32), (38, 16), (20, 8), (0, 4)]
_CHECKPOINT_BELOW_GB = 48


# ----------------------------------------------------------------- reporting
def log(msg: str = "", **kw):
    print(msg, flush=True, **kw)


def rule(title: str):
    log()
    log(f"=== {title} " + "=" * max(0, 68 - len(title)))


class Progress:
    """Single-line progress for someone watching over SSH."""

    def __init__(self, total: int, prefix: str = ""):
        self.total, self.prefix, self.n = total, prefix, 0
        self.t0 = time.time()

    def step(self, **fields):
        self.n += 1
        elapsed = time.time() - self.t0
        rate = self.n / max(elapsed, 1e-9)
        eta = (self.total - self.n) / max(rate, 1e-9)
        bits = " ".join(f"{k}={v}" for k, v in fields.items())
        sys.stdout.write(
            f"\r  {self.prefix} {self.n}/{self.total} "
            f"[{elapsed:5.0f}s elapsed, {eta:5.0f}s left] {bits}   ")
        sys.stdout.flush()

    def done(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


# --------------------------------------------------------------------- setup
def detect_device(requested_batch: int | None):
    """Report the GPU and pick a batch size and a checkpointing decision."""
    if not torch.cuda.is_available():
        log("  no CUDA device — falling back to CPU. This will be very slow, and")
        log("  CPU-vs-GPU latency rows will be identical.")
        return torch.device("cpu"), (requested_batch or 2), False, 0.0

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    batch = requested_batch
    if batch is None:
        batch = next(b for floor, b in _VRAM_BATCH if vram_gb >= floor)
    grad_ckpt = vram_gb < _CHECKPOINT_BELOW_GB

    log(f"  GPU:   {props.name}")
    log(f"  VRAM:  {vram_gb:.1f} GB")
    log(f"  batch: {batch}" + ("" if requested_batch is None else " (from --batch-size)"))
    log(f"  gradient checkpointing: {'on' if grad_ckpt else 'off'}"
        f"  (auto: on below {_CHECKPOINT_BELOW_GB}GB)")
    return torch.device("cuda"), batch, grad_ckpt, vram_gb


def load_model(device, grad_ckpt: bool):
    """Load I-JEPA with eager attention, optionally with checkpointing on.

    ``attn_implementation="eager"`` is not optional: PlasticityScanner reads
    attention weights, and SDPA returns none — routing entropy comes back NaN
    and hub detection reports "no hubs", which reads exactly like a finding.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


# --------------------------------------------------------------------- data
def build_data(sizes: dict, batch_size: int, seed: int):
    """ImageNet-100 as (pixel_values, label) tensors, cached under ~/.cache."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID)
    train_split = ds["train"].shuffle(seed=seed)
    val_split = ds["validation"].shuffle(seed=seed)
    log(f"  dataset: {len(ds['train'])} train / {len(ds['validation'])} val available")

    def encode(split, n, label):
        subset = split.select(range(min(n, len(split))))
        pixels, labels = [], []
        prog = Progress((len(subset) + 63) // 64, f"encoding {label}")
        for i in range(0, len(subset), 64):
            chunk = subset[i:i + 64]
            imgs = [im.convert("RGB") for im in chunk["image"]]
            pixels.append(proc(imgs, return_tensors="pt")["pixel_values"])
            labels.append(torch.tensor(chunk["label"]))
            prog.step()
        prog.done()
        return torch.cat(pixels), torch.cat(labels)

    n_train = sizes["n_train"]
    xtr, ytr = encode(train_split, n_train, "train")
    xpt, ypt = encode(train_split.select(range(n_train, len(train_split))),
                      sizes["n_probe_train"], "probe-train")
    xpv, ypv = encode(val_split, sizes["n_probe_val"], "probe-val")

    n_fi = min(sizes["n_fi"], xpv.shape[0])
    num_classes = int(max(ytr.max(), ypt.max(), ypv.max())) + 1
    loaders = {
        # No shuffling outside training: CKA pairs features example by example.
        "train": DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True),
        "probe_train": DataLoader(TensorDataset(xpt, ypt), batch_size=32),
        "probe_val": DataLoader(TensorDataset(xpv, ypv), batch_size=32),
        "cka": DataLoader(TensorDataset(xpv[:n_fi]), batch_size=16),
    }
    fi_batches = [{"pixel_values": xpv[i:i + 16]} for i in range(0, n_fi, 16)]
    return loaders, fi_batches, num_classes


# ---------------------------------------------------------------- objective
def mask_patches(pixels, patch_size: int, ratio: float):
    """Zero a random ``ratio`` of non-overlapping patches, independently per image."""
    b, c, h, w = pixels.shape
    gh, gw = h // patch_size, w // patch_size
    keep = torch.rand(b, gh * gw, device=pixels.device) >= ratio
    mask = keep.view(b, 1, gh, 1, gw, 1).to(pixels.dtype)
    return (pixels.view(b, c, gh, patch_size, gw, patch_size) * mask).view(b, c, h, w)


def make_step(patch_size: int, use_amp: bool, progress: Progress | None = None,
              restore_masks=None):
    """Build the SALTrainer callback.

    The target pass runs inside ``mask_module.unmasked()`` — a *suspend*, not
    ``deactivate()``, which refills the masks with ones and would throw away the
    pruned set the schedule has accumulated, silently undoing SAL.

    ``restore_masks`` re-applies a checkpointed pruned set on the first step.
    SALTrainer installs the masker inside ``train()``, so the first callback is
    the earliest point at which the masks exist to be written to.
    """
    state = {"restored": restore_masks is None}

    def step(model, batch, optimizer, mask_module):
        if not state["restored"]:
            for layer_idx, saved in restore_masks.items():
                mask_module._masks[int(layer_idx)].copy_(
                    saved.to(mask_module._masks[int(layer_idx)].device))
            state["restored"] = True

        pixels = batch[0]
        visible = mask_patches(pixels, patch_size, PATCH_MASK_RATIO)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_amp else contextlib.nullcontext())
        with autocast:
            with torch.no_grad(), mask_module.unmasked():
                target = model(pixel_values=pixels).last_hidden_state
            predicted = model(pixel_values=visible).last_hidden_state
            loss = torch.nn.functional.mse_loss(predicted, target.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if progress is not None:
            progress.step(loss=f"{loss.item():.5f}",
                          pruned=mask_module.stats["pruned_heads"])
        return loss.item()

    return step


def train_control(model, loader, epochs, patch_size, device, use_amp):
    """No-SAL control: identical objective, optimizer and step count.

    Deliberately not SALTrainer with a near-zero prune_fraction —
    ``num_heads_to_prune`` floors at 1, so "nearly off" is not off. The control
    has to be exactly zero masking.
    """
    class _NoMask:
        def apply_mask(self): pass
        def remove_mask(self): pass
        def unmasked(self): return contextlib.nullcontext()
        stats = {"pruned_heads": 0}

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    losses = []
    for epoch in range(epochs):
        prog = Progress(len(loader), f"control epoch {epoch + 1}/{epochs}")
        step = make_step(patch_size, use_amp, prog)
        total = 0.0
        for batch in loader:
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            total += step(model, batch, opt, _NoMask())
        prog.done()
        losses.append(total / max(len(loader), 1))
        log(f"    epoch {epoch + 1} mean loss {losses[-1]:.6f}")
    return {"losses": losses}


# ------------------------------------------------------------------ pruning
def prune(model, ratio: float, method: str, seed: int = 0):
    """A copy of ``model`` with ``ratio`` of heads masked off (not sliced).

    Masking keeps the parameter count identical across variants, so the table
    isolates *function* from *size*. ``sal.slice_heads()`` is what turns a
    chosen ratio into an actually smaller model.
    """
    import copy

    from sal import SALConfig, arch_support
    from sal.masker import HeadMasker

    pruned = copy.deepcopy(model)
    info = arch_support.detect_architecture(pruned)
    cfg = SALConfig.auto(pruned, prune_fraction=ratio, prune_start_ratio=0.0)
    masker = HeadMasker(pruned, cfg, seed=seed)
    masker.install()

    if method == "random":
        masker.activate()
    elif method == "magnitude":
        masker.apply_mask()
        n_drop = int(round(ratio * info.num_heads))
        projs = arch_support.get_output_projections(pruned, info.attention_pattern)
        for layer_idx, proj in enumerate(projs):
            w = proj.weight.detach()
            per_head = w.view(w.shape[0], info.num_heads, -1).norm(dim=(0, 2))
            for head in torch.argsort(per_head)[:n_drop].tolist():
                masker._masks[layer_idx][head] = 0.0
    else:
        raise ValueError(f"method must be 'random' or 'magnitude', got {method!r}")

    pruned._sal_masker = masker          # keep the hooks alive with the model
    return pruned


def uniform_heads(model, ratio: float, method: str, seed: int = 0) -> list:
    """``(layer, head)`` pairs removing the same number of heads from every layer.

    Slicing needs a uniform count per layer (see ``sal.slicing``), which the
    masker's ``random`` selection does not give: it samples across the whole
    model, so layers lose different numbers of heads. ``random`` here therefore
    samples *within* each layer. ``magnitude`` already drops ``round(ratio *
    heads)`` per layer, so its sliced selection is exactly the masked one.
    """
    import random

    from sal import arch_support

    info = arch_support.detect_architecture(model)
    k = int(round(ratio * info.num_heads))
    rng = random.Random(seed)
    projs = arch_support.get_output_projections(model, info.attention_pattern)
    pairs = []
    for layer_idx, proj in enumerate(projs):
        if method == "random":
            heads = rng.sample(range(info.num_heads), k)
        elif method == "magnitude":
            w = proj.weight.detach()
            per_head = w.view(w.shape[0], info.num_heads, -1).norm(dim=(0, 2))
            heads = torch.argsort(per_head)[:k].tolist()
        else:
            raise ValueError(f"method must be 'random' or 'magnitude', got {method!r}")
        pairs.extend((layer_idx, h) for h in heads)
    return pairs


def slice_variant(model, ratio: float, method: str, seed: int, verify_batch):
    """A physically smaller copy of ``model``, checked against its masked twin.

    Returns ``(sliced, info)``. The check runs the unsliced model with the same
    heads zeroed by hook and compares final hidden states; the relative gap is
    recorded rather than trusted.
    """
    from sal.slicing import SlicingError, slice_heads, verify_slicing

    pairs = uniform_heads(model, ratio, method, seed=seed)
    sliced = slice_heads(model, pairs).eval()
    diff = verify_slicing(model, sliced, pairs, verify_batch)
    with torch.no_grad():
        scale = model.eval()(**verify_batch).last_hidden_state.abs().max().item()
    rel = diff / max(scale, 1e-12)
    if rel > 1e-3:
        raise SlicingError(f"sliced model diverges from the masked model: max abs diff "
                           f"{diff:.3g} on outputs of scale {scale:.3g} ({rel:.2g} relative)")
    nl = model.config.num_hidden_layers
    return sliced, {"heads_removed_per_layer": len(pairs) // nl,
                    "heads_per_layer_after": sliced.config.num_attention_heads,
                    "fraction_removed": len(pairs) / (nl * model.config.num_attention_heads),
                    "max_abs_diff_vs_masked": diff, "relative_diff_vs_masked": rel}


# -------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, model, optimizer, epoch: int, losses, masks, args,
                    with_optimizer: bool = True):
    """Write the checkpoint atomically, or not at all.

    ``torch.save`` straight to the destination leaves a truncated file behind if
    it runs out of space mid-write — and a truncated checkpoint is worse than no
    checkpoint, because ``--resume`` will pick it up and fail somewhere less
    obvious. Writing to a sibling temp file and renaming means the destination
    only ever holds a complete checkpoint; a failed write leaves the previous
    epoch's intact.

    ViT-H/14 is ~2.5GB of fp32 weights and AdamW carries two more moments on top,
    so a full checkpoint is ~7.6GB per epoch. ``with_optimizer=False`` drops it to
    2.5GB, at the cost of resuming with a cold optimizer.
    """
    payload = {
        "epoch": epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": (optimizer.state_dict()
                            if (optimizer and with_optimizer) else None),
        "losses": losses,
        "masks": {str(k): v.detach().cpu() for k, v in masks.items()},
        "args": vars(args),
        "model_id": MODEL_ID,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        size = _free_gb(path.parent)
        raise RuntimeError(
            f"Could not write the checkpoint to {path} ({e}). "
            f"{size:.1f}GB free on that filesystem; a full checkpoint needs about "
            f"{_checkpoint_gb(model, with_optimizer):.1f}GB. Point --output at a "
            "filesystem with room (a network volume's quota is often far smaller "
            "than its reported free space)."
        ) from e
    log(f"    checkpoint -> {path} (epoch {epoch}, "
        f"{path.stat().st_size / 1e9:.1f}GB)")


def _free_gb(directory: Path) -> float:
    try:
        st = os.statvfs(directory)
        return st.f_bavail * st.f_frsize / 1e9
    except (OSError, AttributeError):
        return float("nan")


def _checkpoint_gb(model, with_optimizer: bool = True) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * (12 if with_optimizer else 4) / 1e9


def check_output_space(outdir: Path, model, with_optimizer: bool = True):
    """Fail before training rather than after it.

    A quota is not visible in ``df``: a RunPod network volume reports hundreds of
    terabytes free while refusing to write a single gigabyte. So this actually
    writes a probe file of the size a checkpoint will need.
    """
    need = _checkpoint_gb(model, with_optimizer) + 2.5    # + save_pretrained
    free = _free_gb(outdir)
    log(f"  output space: {free:.1f}GB reported free, ~{need:.1f}GB needed")

    probe = outdir / ".space_probe"
    chunk, written = bytes(64 * 1024 * 1024), 0
    try:
        with open(probe, "wb") as f:
            for _ in range(int(need * 1e9) // len(chunk) + 1):
                f.write(chunk)
                written += len(chunk)
        log("  space probe: OK")
    except OSError as e:
        raise RuntimeError(
            f"Cannot reserve {need:.1f}GB under {outdir} - wrote "
            f"{written / 1e9:.1f}GB then hit: {e}. df reports {free:.1f}GB free, "
            "so this is a quota, not a full disk. Use --output on a filesystem "
            "with real room."
        ) from e
    finally:
        probe.unlink(missing_ok=True)


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    log(f"  resuming from {path}: epoch {ckpt['epoch']}, "
        f"{len(ckpt.get('masks', {}))} layers of mask state")
    return ckpt


def remap_window(cfg, epochs_done: int, total_epochs: int):
    """Shift the prune window into the remaining fraction of a resumed run.

    A resumed run is a shorter run, so the original [start, end] fractions no
    longer point at the same absolute steps. Mapping them keeps the ramp on the
    schedule it would have followed uninterrupted; the already-pruned heads are
    restored separately and count toward each step's target, since
    ``_prune_to_count`` only ever adds.
    """
    done = epochs_done / max(total_epochs, 1)
    remaining = max(1e-9, 1.0 - done)
    start = min(0.98, max(0.0, (cfg.prune_start_ratio - done) / remaining))
    end = min(1.0, max(start + 0.01, (cfg.prune_end_ratio - done) / remaining))
    cfg.prune_start_ratio, cfg.prune_end_ratio = start, end
    return cfg


# ------------------------------------------------------------------ scoring
def score(model, reference, loaders, fi_batches, num_classes, latency_runs,
          device, image_size=224, want_fi=True):
    """Every metric for one variant.

    ``image_size`` comes from the model config rather than a 224 constant: a
    latency probe of the wrong shape throws, and the row would silently read
    ``n/a`` as though the measurement were unavailable rather than misconfigured.
    """
    from sal import (cka_similarity, count_params, knn_accuracy, linear_probe,
                     measure_latency)
    from sal.scanner import FIScanner

    row = {
        "linear_probe": linear_probe(model, loaders["probe_train"],
                                     loaders["probe_val"], num_classes=num_classes),
        "knn_accuracy": knn_accuracy(model, loaders["probe_train"],
                                     loaders["probe_val"], k=20),
        "cka_similarity": (1.0 if model is reference
                           else cka_similarity(reference, model, loaders["cka"])),
        "params": count_params(model),
    }

    image_shape = (1, 3, image_size, image_size)
    for dev in (["cuda", "cpu"] if device.type == "cuda" else ["cpu"]):
        key = f"latency_{'gpu' if dev == 'cuda' else 'cpu'}_ms"
        try:
            # A separate forward pass per device, as asked; measure_latency moves
            # the model and puts it back where it found it.
            row[key] = measure_latency(model, image_shape, device=dev,
                                       warmup=max(2, latency_runs // 10),
                                       runs=latency_runs)
        except Exception as e:
            row[key] = None
            row.setdefault("skipped", {})[key] = f"{type(e).__name__}: {e}"
    row.setdefault("latency_gpu_ms", None)

    if want_fi:
        try:
            row["fi_score"] = FIScanner(model, fi_batches,
                                        num_samples=len(fi_batches) * 16).scan().fi_score
        except Exception as e:
            row["fi_score"] = None
            row.setdefault("skipped", {})["fi_score"] = f"{type(e).__name__}: {e}"
    return row


# ------------------------------------------------------------------ figures
def make_figures(outdir: Path, results: dict, fi_before: float, fi_after: float,
                 original, sal_model, image):
    """Write the four PNGs. A failure here must not lose the metrics."""
    figs = outdir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    written = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not installed — skipping figures "
            "(metrics are already saved).")
        return written

    from sal.visualization import compare_feature_maps, visualize_compression_impact

    def attempt(name, fn):
        try:
            fn()
            written.append(name)
            log(f"    {name}")
        except Exception as e:
            log(f"    {name} FAILED: {type(e).__name__}: {e}")
        finally:
            plt.close("all")

    attempt("feature_maps_comparison.png", lambda: compare_feature_maps(
        original, sal_model, image, num_features=8,
        labels=("pretrained", "SAL-trained"),
        save_path=str(figs / "feature_maps_comparison.png")))

    quality = {k: v for k, v in results.items() if k != "original"}
    attempt("compression_table.png", lambda: visualize_compression_impact(
        quality, metrics=["linear_probe", "knn_accuracy", "cka_similarity"],
        save_path=str(figs / "compression_table.png"),
        title="Representation quality after pruning"))

    def fi_chart():
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["before SAL", "after SAL"], [fi_before, fi_after],
               color=["#757575", "#2e7d32"])
        for i, v in enumerate([fi_before, fi_after]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
        ax.set_ylabel("Fragility Index (lower = more redundant)")
        ax.set_title("Structural fragility, before and after SAL training")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / "fi_before_after.png", dpi=150, bbox_inches="tight")

    attempt("fi_before_after.png", fi_chart)

    lat = {k: v for k, v in results.items()
           if isinstance(v.get("latency_cpu_ms"), (int, float))}
    attempt("latency_comparison.png", lambda: visualize_compression_impact(
        lat, metrics=["latency_cpu_ms"],
        save_path=str(figs / "latency_comparison.png"),
        title="Inference latency (median ms)"))
    return written


# ------------------------------------------------------------------- table
def _box_chars():
    """Box-drawing glyphs, or ASCII when the terminal cannot encode them.

    The summary is the last thing printed after hours of compute. A console
    using cp1252 (or any non-UTF-8 encoding) raises UnicodeEncodeError on the
    box characters, which would throw away the whole run's output at the final
    step. ASCII is uglier and always prints.
    """
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "┌─│".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return dict(h="-", v="|", tl="+", tr="+", bl="+", br="+",
                    lt="+", rt="+", tt="+", bt="+", x="+")
    return dict(h="─", v="│", tl="┌", tr="┐", bl="└",
                br="┘", lt="├", rt="┤", tt="┬",
                bt="┴", x="┼")


def summary_table(results: dict, order: list) -> str:
    # Width fits the longest label the script generates ("ctrl+magnitude-33%"
    # plus padding); a narrower column overflows and breaks every border below it.
    width = max(17, max((len(n) for n in order), default=0) + 3)
    cols = [("Variant", width), ("Params", 8), ("Lin.Probe", 11), ("kNN", 11),
            ("CKA", 11), ("GPU ms", 10), ("CPU ms", 10)]
    b = _box_chars()
    top = b["tl"] + b["tt"].join(b["h"] * w for _, w in cols) + b["tr"]
    mid = b["lt"] + b["x"].join(b["h"] * w for _, w in cols) + b["rt"]
    bot = b["bl"] + b["bt"].join(b["h"] * w for _, w in cols) + b["br"]

    def row(cells):
        return (b["v"] + b["v"].join(f" {c:<{w - 2}} "
                                     for c, (_, w) in zip(cells, cols)) + b["v"])

    def pct(v):
        return "  n/a" if v is None else f"{v * 100:.1f}%"

    def ms(v):
        return "n/a" if v is None else f"{v:.0f}ms"

    lines = [top, row([c for c, _ in cols]), mid]
    for name in order:
        r = results.get(name)
        if not r:
            continue
        params = r.get("params")
        lines.append(row([name, "n/a" if params is None else f"{params / 1e6:.0f}M",
                          pct(r["linear_probe"]), pct(r["knn_accuracy"]),
                          f"{r['cka_similarity']:.3f}",
                          ms(r.get("latency_gpu_ms")), ms(r.get("latency_cpu_ms"))]))
    lines.append(bot)
    return "\n".join(lines)


# -------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SAL compression benchmark on I-JEPA ViT-H/14",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 100 images — validates the path, means nothing")
    p.add_argument("--epochs", type=int, default=None,
                   help="default: 5, or 1 under --smoke. An explicit value "
                        "always wins, including with --smoke.")
    p.add_argument("--mask-ratio", type=float, default=0.3,
                   help="fraction of attention heads SAL prunes during training")
    p.add_argument("--output", default="results/", help="output directory")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override the VRAM-based auto-detect")
    p.add_argument("--resume", default=None, help="path to checkpoint.pt")
    p.add_argument("--control", action="store_true",
                   help="also train a no-SAL arm — the comparison that actually "
                        "tests SAL training rather than head selection")
    p.add_argument("--no-space-check", action="store_true",
                   help="skip the pre-training probe that reserves checkpoint space")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=MODEL_ID,
                   help=f"Hub id of a ViT-family encoder (default: {MODEL_ID}); "
                        "e.g. facebook/dinov2-large")
    p.add_argument("--prune-ratios", type=float, nargs="+", default=list(PRUNE_RATIOS),
                   help="post-training pruning ratios to benchmark (default: 0.33 0.5)")
    p.add_argument("--slice", action="store_true",
                   help="also physically remove the heads with slice_heads() and "
                        "score the smaller models — the rows with real latency")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # load_model/build_data read the module-level id, so they keep their
    # signatures (the tests replace them) while following --model.
    global MODEL_ID
    MODEL_ID = args.model
    started = time.time()
    torch.manual_seed(args.seed)

    sizes = dict(n_train=N_TRAIN, n_probe_train=N_PROBE_TRAIN,
                 n_probe_val=N_PROBE_VAL, n_fi=N_FI_PROBE,
                 latency_runs=LATENCY_RUNS, epochs=DEFAULT_EPOCHS)
    if args.smoke:
        sizes.update(SMOKE)
    # An explicit --epochs beats the smoke default; otherwise `--smoke --epochs 3`
    # would quietly run one epoch, which is exactly the kind of silent override
    # that makes a resumed run look finished when it is not.
    if args.epochs is not None:
        sizes["epochs"] = args.epochs
    epochs = sizes["epochs"]
    args.epochs = epochs

    outdir = Path(args.output)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "model").mkdir(parents=True, exist_ok=True)

    rule("Environment")
    device, batch_size, grad_ckpt, vram = detect_device(args.batch_size)
    log(f"  output: {outdir.resolve()}")
    log(f"  mode:   {'SMOKE (1 epoch, 100 images)' if args.smoke else 'full'}"
        f" | epochs={epochs} | mask_ratio={args.mask_ratio}")
    use_amp = device.type == "cuda"

    rule("Assets")
    log(f"  model:   {MODEL_ID}")
    log(f"  dataset: {DATASET_ID}")
    log("  (cached under ~/.cache/huggingface — re-runs do not re-download)")
    model = load_model(device, grad_ckpt)
    patch_size = model.config.patch_size
    from sal import count_params
    log(f"  loaded:  {count_params(model) / 1e6:.0f}M params, "
        f"{model.config.num_hidden_layers} layers x "
        f"{model.config.num_attention_heads} heads, patch {patch_size}")

    loaders, fi_batches, num_classes = build_data(sizes, batch_size, args.seed)
    fi_batches = [{k: v.to(device) for k, v in b.items()} for b in fi_batches]
    log(f"  classes: {num_classes}")
    # Read the size off the processed images, not the config: DINOv2's config
    # says 518 (its pretraining resolution) while its processor crops to 224.
    image_size = next(iter(loaders["cka"]))[0].shape[-1]

    # 4 ------------------------------------------------------- pre-SAL scan
    rule("Pre-SAL structural scan")
    from sal import PlasticityScanner
    from sal.scanner import FIScanner
    n_scan = len(fi_batches) * 16
    scan_before = FIScanner(model, fi_batches, num_samples=n_scan).scan()
    plast_before = PlasticityScanner(model, fi_batches, num_samples=n_scan).scan()
    log(f"  {scan_before.summary}")
    log(f"  absorption: {plast_before.summary}")
    pre = {"fi_score": scan_before.fi_score,
           "layer_classification": {str(k): v.value
                                    for k, v in scan_before.layer_map.items()},
           "absorption_map": {str(k): v for k, v in plast_before.absorption_map.items()}}
    (outdir / "pre_sal_scan.json").write_text(json.dumps(pre, indent=2))
    log(f"  saved {outdir / 'pre_sal_scan.json'}")

    import copy
    original = copy.deepcopy(model).eval()      # pretrained reference for CKA
    probe_image = next(iter(loaders["cka"]))[0][:1].to(device)

    # 5 -------------------------------------------------------- SAL training
    rule(f"SAL training ({epochs} epochs, mask_ratio={args.mask_ratio})")
    if not args.no_space_check:
        check_output_space(outdir, model)
    from sal import SALConfig
    from sal.trainer import SALTrainer

    cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    restore_masks, epochs_done, losses = None, 0, []
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume))
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        restore_masks = ckpt.get("masks") or None
        epochs_done, losses = ckpt["epoch"], list(ckpt.get("losses", []))
        cfg = remap_window(cfg, epochs_done, epochs)
        log(f"  {epochs - epochs_done} epochs left; prune window remapped to "
            f"[{cfg.prune_start_ratio:.3f}, {cfg.prune_end_ratio:.3f}]")

    remaining = max(0, epochs - epochs_done)
    masker_stats = {}
    if remaining == 0:
        log("  checkpoint is already complete — skipping training.")
    else:
        # The whole run is a single SALTrainer call: one call per epoch would
        # restart the prune schedule each time. The per-epoch checkpoint is
        # therefore written from inside the callback, at each epoch boundary.
        prog = Progress(len(loaders["train"]) * remaining, "training")
        base_step = make_step(patch_size, use_amp, prog, restore_masks)
        batches_per_epoch = len(loaders["train"])
        seen = {"n": 0, "sum": 0.0}

        def checkpointing_step(m, batch, opt, mask_module):
            loss = base_step(m, batch, opt, mask_module)
            seen["n"] += 1
            seen["sum"] += loss
            if seen["n"] % batches_per_epoch == 0:
                epoch_no = epochs_done + seen["n"] // batches_per_epoch
                losses.append(seen["sum"] / batches_per_epoch)
                seen["sum"] = 0.0
                prog.done()
                log(f"    epoch {epoch_no}/{epochs} "
                    f"| loss {losses[-1]:.6f} | masker: {mask_module.stats}")
                save_checkpoint(outdir / "checkpoint.pt", m, opt, epoch_no,
                                losses, mask_module._masks, args)
                prog.t0 = time.time()
            return loss

        trainer = SALTrainer(model, cfg, optimizer, loaders["train"],
                             seed=args.seed, train_step=checkpointing_step)
        history = trainer.train(num_epochs=remaining)
        prog.done()
        masker_stats = history["masker_stats"]
        log(f"  losses: {[round(x, 6) for x in losses]}")
        log(f"  masker: {masker_stats}")

    # 6 ------------------------------------------------------- post-SAL scan
    rule("Post-SAL structural scan")
    scan_after = FIScanner(model, fi_batches, num_samples=n_scan).scan()
    plast_after = PlasticityScanner(model, fi_batches, num_samples=n_scan).scan()
    delta = scan_after.fi_score - scan_before.fi_score
    log(f"  FI {scan_before.fi_score:.4f} -> {scan_after.fi_score:.4f} "
        f"({delta:+.4f})")
    log(f"  {scan_after.summary}")
    post = {"fi_score": scan_after.fi_score, "fi_delta": delta,
            "layer_classification": {str(k): v.value
                                     for k, v in scan_after.layer_map.items()},
            "absorption_map": {str(k): v for k, v in plast_after.absorption_map.items()}}
    (outdir / "post_sal_scan.json").write_text(json.dumps(post, indent=2))
    log(f"  saved {outdir / 'post_sal_scan.json'}")

    # optional control arm ---------------------------------------------------
    control_model, control_hist = None, None
    if args.control:
        rule(f"Control arm ({epochs} epochs, no head masking)")
        control_model = load_model(device, grad_ckpt)
        control_hist = train_control(control_model, loaders["train"], epochs,
                                     patch_size, device, use_amp)

    # 7 ---------------------------------------------------------- benchmark
    rule("Compression benchmark")
    lat_runs = sizes["latency_runs"]
    results, order = {}, []

    def add(name, m, reference):
        log(f"  scoring {name} ...")
        results[name] = score(m, reference, loaders, fi_batches, num_classes,
                              lat_runs, device, image_size=image_size)
        order.append(name)

    slicing, slicing_errors = {}, {}
    verify_batch = fi_batches[0]

    def pruned_arm(prefix, trained):
        for ratio in args.prune_ratios:
            for method in SELECTION_METHODS:
                pruned = prune(trained, ratio, method, seed=args.seed)
                add(f"{prefix}+{method}-{int(round(ratio * 100))}%", pruned, trained)
                del pruned
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if not args.slice:
                continue
            for method in SELECTION_METHODS:
                label = "random-uniform" if method == "random" else method
                name = f"{prefix}+{label}-{int(round(ratio * 100))}%-sliced"
                try:
                    sliced, info = slice_variant(trained, ratio, method, args.seed,
                                                 verify_batch)
                except Exception as e:
                    slicing_errors[name] = f"{type(e).__name__}: {e}"
                    log(f"  SLICING FAILED for {name}: {slicing_errors[name]}")
                    continue
                add(name, sliced, trained)
                slicing[name] = info
                del sliced
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    add("original", original, original)
    add("sal-trained", model, original)
    pruned_arm("sal", model)

    if control_model is not None:
        add("control-trained", control_model, original)
        pruned_arm("ctrl", control_model)

    base = results["original"]
    for name, info in slicing.items():
        r = results[name]
        info["params"] = r["params"]
        info["param_fraction"] = r["params"] / base["params"]
        for dev in ("gpu", "cpu"):
            ms, ref = r.get(f"latency_{dev}_ms"), base.get(f"latency_{dev}_ms")
            info[f"speedup_{dev}"] = (ref / ms) if (ms and ref) else None

    # 8 -------------------------------------------------------------- save
    rule("Saving")
    model.save_pretrained(outdir / "model", safe_serialization=True)
    log(f"  model -> {outdir / 'model'} (config.json + model.safetensors)")

    figures = make_figures(outdir, results, scan_before.fi_score,
                           scan_after.fi_score, original, model, probe_image)

    payload = {
        "model": MODEL_ID, "dataset": DATASET_ID, "smoke": args.smoke,
        "seed": args.seed, "epochs": epochs, "mask_ratio": args.mask_ratio,
        "patch_mask_ratio": PATCH_MASK_RATIO, "batch_size": batch_size,
        "gradient_checkpointing": grad_ckpt, "vram_gb": round(vram, 1),
        "device": str(device),
        "fi_caveat": "FI is comparable only between models with the same head "
                     "count. Masked heads emit zero signatures, carry no edges, "
                     "and push the fixed edge budget onto survivors, which lowers "
                     "FI by construction. Use slice_heads() to compare sizes.",
        "latency_caveat": "Rows without '-sliced' mask rather than slice, so "
                          "parameter count and FLOPs are unchanged and they are "
                          "marginally SLOWER from hook overhead. '-sliced' rows "
                          "physically remove the heads (slice_heads()); only they "
                          "show real size and latency. Latency is batch-1 median.",
        "prune_ratios": args.prune_ratios,
        "slice_caveat": ("'-sliced' rows remove round(ratio * heads) heads from EVERY "
                         "layer. 'magnitude' selects exactly as its masked row; "
                         "'random-uniform' samples within each layer, unlike the "
                         "masked 'random' row, which samples across the whole model. "
                         "Each sliced model is checked against the same heads masked "
                         "by hook (relative_diff_vs_masked)."
                         if args.slice else None),
        "slicing": slicing, "slicing_errors": slicing_errors,
        "objective": "I-JEPA-shaped: predict the clean-image representation from "
                     "a patch-masked image. No predictor network, no EMA target "
                     "encoder — NOT I-JEPA pretraining, and for DINOv2 NOT its "
                     "self-distillation objective: the same masked-prediction "
                     "objective is applied to every --model so arms stay comparable.",
        "comparison": ("SAL-training vs no-SAL control (--control)" if args.control
                       else "head SELECTION only (random vs magnitude) on one "
                            "SAL-trained model — pass --control to compare "
                            "SAL training against a no-SAL arm"),
        "num_classes": num_classes,
        "fi_before": scan_before.fi_score, "fi_after": scan_after.fi_score,
        "sal_losses": losses, "masker_stats": masker_stats,
        "control_losses": control_hist["losses"] if control_hist else None,
        "results": results, "figures": figures,
        "wall_clock_s": round(time.time() - started, 1),
    }
    (outdir / "jepa_sal_benchmark.json").write_text(json.dumps(payload, indent=2))
    log(f"  metrics -> {outdir / 'jepa_sal_benchmark.json'}")

    # 9 ------------------------------------------------------------ summary
    rule("Summary")
    log(summary_table(results, order))
    log()
    log(f"FI: {scan_before.fi_score:.4f} -> {scan_after.fi_score:.4f} ({delta:+.4f})")
    log(f"wall clock: {(time.time() - started) / 60:.1f} min")
    log()
    log("CKA is measured against each arm's own unpruned model.")
    if not args.control:
        log("This table compares head SELECTION (random vs magnitude) on one")
        log("SAL-trained model. It does NOT show whether SAL training helped —")
        log("for that, re-run with --control.")
    if args.smoke:
        log("SMOKE RUN — 100 images, 1 epoch. These numbers mean nothing.")
    log(f"Everything is in {outdir.resolve()} — scp that directory back.")
    return payload


if __name__ == "__main__":
    main()
