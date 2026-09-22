#!/usr/bin/env python3
"""SAL + QAT prototype: are pruning resilience and quantization resilience additive?

Trains four variants of ViT-B/16 on an ImageNet-100 subset, identical in every
respect except the two switches:

  1. Baseline    — plain fine-tuning (no SAL, no QAT)
  2. SAL only    — progressive random head masking during training
  3. QAT only    — fake-quantized (4-bit) weights during training
  4. SAL + QAT   — both at once

then scores each one's top-1 accuracy under:

  FP16      no compression
  Prune33%  a third of the heads physically removed with ``slice_heads()``
  Prune50%  half of the heads removed
  INT8      8-bit weights
  INT4      4-bit weights
  P50+I4    half the heads removed, *then* 4-bit weights — the full pipeline

The bottom-right cell (SAL + QAT at P50+I4) against the baseline's is the
number this prototype exists to produce.

Usage::

    python scripts/run_sal_qat_prototype.py --smoke          # < 5 min, validates the path
    python scripts/run_sal_qat_prototype.py --epochs 5 --output results/

Requirements::

    bash scripts/setup_qat_prototype.sh     # or: pip install -e ".[dev,reports]"
                                            #     transformers datasets torchvision

GPU: an A10G (24GB) is plenty — ViT-B/16 is 86M parameters.

READ THIS BEFORE QUOTING ANY NUMBER
-----------------------------------

**ViT-B/16 comes from transformers, not timm.** ``google/vit-base-patch16-224``
(ImageNet-21k pretrained, ImageNet-1k fine-tuned) with a fresh 100-way head.
timm's ViT keeps its attention in ``blocks.N.attn.{qkv,proj}``, which sal-torch's
architecture registry does not know, so ``HeadMasker``, ``slice_heads()`` and
``FIScanner`` would all refuse it. HF's ViT is registered and validated.

**QAT targets 4-bit weights, not the INT8 default.** The key cell is INT4, and
INT8 weight rounding is already near-lossless without any training — an INT8 QAT
arm would be a control, not a treatment. Training uses native ``torch.ao``
QAT (``prepare_qat``) with a per-channel symmetric 4-bit weight fake-quantizer
and no activation quantization. If ``prepare_qat`` fails, the script falls back
to a weight parametrization with a straight-through estimator. The fallback is
recorded in the JSON as ``qat_backend``.

**The quantized columns are weight round-trips, not kernels.** INT8 and INT4
accuracy come from quantizing and dequantizing every ``nn.Linear`` weight
(per-output-channel symmetric, the classifier head excluded). That is exactly
the grid QAT trains against, it is deterministic, and it runs on the GPU. When
bitsandbytes is installed, the INT4 cells are cross-checked against real NF4
(``int4_nf4`` in the JSON). NF4 is a different grid from the uniform one QAT
learns, so a QAT gain that shows up in INT4 and disappears in NF4 does not
transfer to that backend.

**Size is computed, not measured**: fp16 for everything that is not quantized,
``bits/8`` bytes per quantized weight plus one fp16 scale per output channel.

**CPU latency is measured once per compression setting**, not per variant.
All four variants have the same architecture, so they have the same latency.
INT8 latency is a real ``torch.ao`` dynamic-INT8 model. Stock PyTorch has no CPU
INT4 kernel, so INT4 latency is ``n/a``: a simulated INT4 model runs at fp32
speed, and reporting that number would imply a speedup that isn't there.

**Pruning removes random heads, the same number from every layer.** Each
setting is repeated over ``--prune-seeds`` head selections, and every variant
sees the same selections. The table shows the mean. ``random`` is sal-torch's
validated default. ``magnitude`` loses 2.7-5.7pp on GPT-2 Medium, see README.

**One training seed.** All four arms train with the same seed, data order and
step count. That makes them comparable, but it doesn't make the result
repeatable. Treat a sub-1pp gap as "not shown" until ``--seed`` has been varied.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

MODEL_ID = "google/vit-base-patch16-224"
DATASET_ID = "clane9/imagenet-100"
IMAGE_SIZE = 224

VARIANTS = (                      # (key, label, use_sal, use_qat)
    ("baseline", "Baseline", False, False),
    ("sal", "SAL only", True, False),
    ("qat", "QAT only", False, True),
    ("sal_qat", "SAL + QAT", True, True),
)
CONFIGS = ("fp16", "prune33", "prune50", "int8", "int4", "prune50_int4")
CONFIG_LABELS = {"fp16": "FP16", "prune33": "Prune33%", "prune50": "Prune50%",
                 "int8": "INT8", "int4": "INT4", "prune50_int4": "P50+I4"}
KEY_CELL = "prune50_int4"

LR_BACKBONE = 5e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_FRACTION = 0.05
MASK_RATIO = 0.3                  # fraction of heads SAL prunes by end of training
QAT_BITS = 4
SKIP_QUANT = ("classifier",)      # the output head stays in full precision

# Full-run sizes; --smoke overrides them. ImageNet-100 has 1300 train images per
# class; 200 per class keeps four arms x five epochs well inside the GPU budget.
N_TRAIN = 20_000
N_VAL = 5_000
N_FI = 64
LATENCY_RUNS_CPU = 20
LATENCY_RUNS_GPU = 50
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 64
SMOKE = dict(n_train=200, n_val=100, n_fi=16, epochs=1,
             latency_runs_cpu=3, latency_runs_gpu=5, prune_seeds=1)


# ----------------------------------------------------------------- reporting
def log(msg: str = ""):
    print(msg, flush=True)


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
        eta = (self.total - self.n) * elapsed / max(self.n, 1)
        bits = " ".join(f"{k}={v}" for k, v in fields.items())
        sys.stdout.write(f"\r  {self.prefix} {self.n}/{self.total} "
                         f"[{elapsed:5.0f}s elapsed, {eta:5.0f}s left] {bits}   ")
        sys.stdout.flush()

    def done(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


# --------------------------------------------------------------------- setup
def detect_device():
    if not torch.cuda.is_available():
        log("  no CUDA device — running on CPU. Fine for --smoke, far too slow otherwise.")
        return torch.device("cpu"), 0.0
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    log(f"  GPU:  {props.name}")
    log(f"  VRAM: {vram:.1f} GB")
    return torch.device("cuda"), vram


def load_model(num_classes: int):
    """ViT-B/16 with a fresh ``num_classes``-way head."""
    from transformers import ViTForImageClassification

    return ViTForImageClassification.from_pretrained(
        MODEL_ID, num_labels=num_classes, ignore_mismatched_sizes=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- data
def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,3,H,W] -> the ViT processor's float input (mean 0.5, std 0.5)."""
    return (x_uint8.float() / 255.0 - 0.5) / 0.5


def build_data(sizes: dict, seed: int, workers: int = 8):
    """ImageNet-100 subsets as uint8 tensors held in memory.

    Images are resized to 224x224 once, up front, and kept as uint8 (150KB each),
    so every arm trains on byte-identical inputs and no epoch waits on JPEG
    decoding. Returns ``(train, val, num_classes)`` with each split an
    ``(images_uint8, labels)`` pair.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    ds = load_dataset(DATASET_ID)
    log(f"  dataset: {len(ds['train'])} train / {len(ds['validation'])} val available")

    def collate(rows):
        imgs = [r["image"].convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), 2)  # bilinear
                for r in rows]
        x = torch.stack([torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
                         .view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1) for im in imgs])
        return x, torch.tensor([r["label"] for r in rows])

    def encode(split, n, label):
        subset = split.shuffle(seed=seed).select(range(min(n, len(split))))
        dl = DataLoader(subset, batch_size=64, num_workers=workers, collate_fn=collate)
        xs, ys = [], []
        prog = Progress(len(dl), f"encoding {label}")
        for x, y in dl:
            xs.append(x)
            ys.append(y)
            prog.step()
        prog.done()
        return torch.cat(xs), torch.cat(ys)

    train = encode(ds["train"], sizes["n_train"], "train")
    val = encode(ds["validation"], sizes["n_val"], "val")
    num_classes = len(ds["train"].features["label"].names)
    return train, val, num_classes


def batches(split, batch_size: int, shuffle: bool = False, seed: int = 0):
    """A DataLoader over an in-memory ``(uint8 images, labels)`` split.

    Shuffling uses its own seeded generator, so every arm sees the same order.
    """
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(*split), batch_size=batch_size,
                      shuffle=shuffle, generator=g)


# ---------------------------------------------------------------- quantization
def _is_quantizable(name: str, mod: nn.Module) -> bool:
    return isinstance(mod, nn.Linear) and not (set(name.split(".")) & set(SKIP_QUANT))


def _qrange(bits: int):
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def fake_quantize_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric quantize/dequantize.

    Same formula as torch.ao's ``per_channel_symmetric`` fake-quantizer:
    ``scale = absmax / ((qmax - qmin) / 2)``. The PTQ columns and the QAT arms
    therefore use exactly the same grid.
    """
    qmin, qmax = _qrange(bits)
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / ((qmax - qmin) / 2)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / scale), qmin, qmax)
    return (q * scale).reshape(w.shape)


def quantize_weights(model: nn.Module, bits: int) -> nn.Module:
    """A copy of ``model`` with every eligible Linear weight rounded to ``bits``."""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for name, mod in q.named_modules():
            if _is_quantizable(name, mod):
                mod.weight.copy_(fake_quantize_weight(mod.weight.float(), bits)
                                 .to(mod.weight.dtype))
    return q


def size_mb(model: nn.Module, bits: int | None) -> float:
    """Deployed size in MB: fp16 everywhere, ``bits`` for quantized Linear weights."""
    total = sum(p.numel() for p in model.parameters())
    if bits is None:
        return total * 2 / 1e6
    qw = channels = 0
    for name, mod in model.named_modules():
        if _is_quantizable(name, mod):
            qw += mod.weight.numel()
            channels += mod.weight.shape[0]
    return ((total - qw) * 2 + qw * bits / 8 + channels * 2) / 1e6


# -------------------------------------------------------------------------- QAT
class _STEFakeQuant(nn.Module):
    """Parametrization: forward sees quantized weights, gradients pass straight through."""

    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits

    def forward(self, w):
        return w + (fake_quantize_weight(w, self.bits) - w).detach()


def setup_qat(model: nn.Module, bits: int = QAT_BITS, force_manual: bool = False):
    """Make ``model`` quantization-aware for ``bits``-bit weights.

    Tries native torch.ao QAT first: ``prepare_qat`` swaps each eligible
    ``nn.Linear`` for a ``torch.ao.nn.qat.Linear`` whose weight passes through a
    per-channel symmetric fake-quantizer. Activations are not quantized, because
    none of the deployment targets here quantize them. If that fails, the
    fallback is a weight parametrization with a straight-through estimator. It
    uses the same grid, but scales from the exact absmax rather than a moving
    average.

    Returns ``(model, backend)``, where ``backend`` is ``"torch.ao"`` or
    ``"manual-ste"``.
    """
    if not force_manual:
        try:
            import torch.ao.quantization as tq

            qmin, qmax = _qrange(bits)
            weight_fq = tq.FakeQuantize.with_args(
                observer=tq.MovingAveragePerChannelMinMaxObserver,
                quant_min=qmin, quant_max=qmax, dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric, ch_axis=0)
            qconfig = tq.QConfig(
                activation=tq.PlaceholderObserver.with_args(dtype=torch.float),
                weight=weight_fq)
            for name, mod in model.named_modules():
                if _is_quantizable(name, mod):
                    mod.qconfig = qconfig
            model.train()
            prepared = tq.prepare_qat(model, inplace=False)
            n = sum(isinstance(m, torch.ao.nn.qat.Linear) for m in prepared.modules())
            if n == 0:
                raise RuntimeError("prepare_qat swapped no Linear layers")
            log(f"  QAT: torch.ao prepare_qat, {n} Linear layers at {bits}-bit weights")
            return prepared, "torch.ao"
        except Exception as e:  # noqa: BLE001 — the fallback exists for exactly this
            log(f"  QAT: torch.ao prepare_qat failed ({type(e).__name__}: {e}); "
                "falling back to the manual straight-through fake-quantizer")
            for mod in model.modules():
                if hasattr(mod, "qconfig"):
                    del mod.qconfig

    from torch.nn.utils import parametrize

    n = 0
    for name, mod in model.named_modules():
        if _is_quantizable(name, mod):
            parametrize.register_parametrization(mod, "weight", _STEFakeQuant(bits))
            n += 1
    log(f"  QAT: manual STE parametrization, {n} Linear layers at {bits}-bit weights")
    return model, "manual-ste"


def strip_qat(model: nn.Module) -> nn.Module:
    """Return ``model`` as plain ``nn.Linear`` layers holding the latent float weights.

    Every variant has to go through the same post-training compression from
    ordinary float weights. ``slice_heads()`` also can't narrow a QAT Linear:
    the per-channel observer buffers would keep their old width.
    """
    from torch.nn.utils import parametrize

    for mod in model.modules():
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)

    qat_linear = getattr(getattr(torch.ao.nn, "qat", None), "Linear", None)

    def walk(parent):
        for name, child in list(parent.named_children()):
            if qat_linear is not None and isinstance(child, qat_linear):
                lin = nn.Linear(child.in_features, child.out_features,
                                bias=child.bias is not None)
                lin.weight = nn.Parameter(child.weight.detach().clone())
                if child.bias is not None:
                    lin.bias = nn.Parameter(child.bias.detach().clone())
                setattr(parent, name, lin.to(child.weight.device))
            else:
                walk(child)
            for attr in ("qconfig", "activation_post_process"):
                if hasattr(child, attr) and not isinstance(child, qat_linear or ()):
                    with contextlib.suppress(AttributeError):
                        delattr(child, attr)

    walk(model)
    return model


# ------------------------------------------------------------------- training
class _NoMask:
    """Stand-in masker for the no-SAL arms: same step function, nothing masked."""

    def apply_mask(self): pass
    def remove_mask(self): pass
    def unmasked(self): return contextlib.nullcontext()
    stats = {"pruned_heads": 0}


def make_step(device, use_amp: bool, progress: Progress | None = None):
    """The one training step every arm runs: supervised cross-entropy.

    SAL arms get masking from SALTrainer, which advances the prune schedule and
    leaves masking on before calling this. QAT arms get fake quantization from
    their modules. Neither needs anything from the step itself, which is why all
    four arms can share it.
    """
    def step(model, batch, optimizer, mask_module):
        x, y = batch[0].to(device), batch[1].to(device)
        x = normalize(x)
        flip = torch.rand(x.shape[0], device=device) < 0.5          # horizontal flip
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_amp else contextlib.nullcontext())
        with autocast:
            logits = model(pixel_values=x).logits
        loss = nn.functional.cross_entropy(logits.float(), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if progress is not None:
            progress.step(loss=f"{loss.item():.4f}",
                          pruned=mask_module.stats["pruned_heads"])
        return loss.item()

    return step


def make_optimizer(model, total_steps: int):
    """AdamW, a 10x higher LR for the fresh head, linear warmup then cosine decay."""
    head, body = [], []
    for name, p in model.named_parameters():
        (head if name.split(".")[0] == "classifier" else body).append(p)
    opt = torch.optim.AdamW([{"params": body, "lr": LR_BACKBONE},
                             {"params": head, "lr": LR_HEAD}],
                            weight_decay=WEIGHT_DECAY)
    warmup = max(1, int(WARMUP_FRACTION * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def train_variant(key, use_sal, use_qat, num_classes, train_split, epochs, batch_size,
                  seed, device, use_amp, mask_ratio, force_manual_qat=False):
    """Build and train one variant. Returns ``(model, info)``, with QAT stripped."""
    from sal import SALConfig, SALTrainer

    set_seed(seed)                     # identical head init across arms
    model = load_model(num_classes)
    qat_backend = None
    if use_qat:
        model, qat_backend = setup_qat(model, QAT_BITS, force_manual=force_manual_qat)
    model.to(device).train()

    loader = batches(train_split, batch_size, shuffle=True, seed=seed)
    total = len(loader) * epochs
    optimizer, scheduler = make_optimizer(model, total)
    prog = Progress(total, f"{key} training")
    step = make_step(device, use_amp, prog)

    t0 = time.time()
    if use_sal:
        cfg = SALConfig.auto(model, prune_fraction=mask_ratio)
        trainer = SALTrainer(model, cfg, optimizer, loader, scheduler=scheduler,
                             seed=seed, train_step=step)
        history = trainer.train(num_epochs=epochs)
        losses, masker_stats = history["losses"], history["masker_stats"]
    else:
        # Not SALTrainer with a tiny prune_fraction: num_heads_to_prune floors at
        # 1, so "nearly off" is not off. The control has to have no masking at all.
        losses, no_mask = [], _NoMask()
        for _ in range(epochs):
            total_loss = 0.0
            for batch in loader:
                total_loss += step(model, batch, optimizer, no_mask)
                scheduler.step()
            losses.append(total_loss / max(len(loader), 1))
        masker_stats = None
    prog.done()

    if use_qat:
        model = strip_qat(model)
    model.eval()
    info = {"losses": losses, "masker_stats": masker_stats, "qat_backend": qat_backend,
            "train_seconds": round(time.time() - t0, 1), "steps": total}
    log(f"  {key}: epoch losses {[round(x, 4) for x in losses]}"
        + (f" | masker {masker_stats}" if masker_stats else "")
        + f" | {info['train_seconds']:.0f}s")
    return model, info


# ------------------------------------------------------------------- scoring
@torch.no_grad()
def accuracy(model, val_split, device, use_amp: bool, batch_size: int = 128) -> float:
    """Top-1 accuracy of the model's own classifier on the val split."""
    model.eval()
    dev = next(model.parameters()).device
    correct = n = 0
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_amp and dev.type == "cuda") else contextlib.nullcontext())
    for x, y in batches(val_split, batch_size):
        with autocast:
            logits = model(pixel_values=normalize(x.to(dev))).logits
        correct += int((logits.argmax(-1).cpu() == y).sum())
        n += y.shape[0]
    return correct / max(n, 1)


def uniform_random_heads(num_layers: int, num_heads: int, ratio: float, seed: int):
    """``(layer, head)`` pairs: ``round(ratio * heads)`` random heads from every layer.

    ``slice_heads()`` needs the same count per layer. Depends only on the seed
    and the shape, so every variant loses exactly the same heads.
    """
    k = int(round(ratio * num_heads))
    rng = random.Random(seed)
    return [(layer, h) for layer in range(num_layers)
            for h in sorted(rng.sample(range(num_heads), k))]


def sliced(model, ratio: float, seed: int, verify_batch=None):
    """Physically remove ``ratio`` of the heads; optionally check against masking."""
    from sal import slice_heads
    from sal.slicing import SlicingError, verify_slicing

    cfg = model.config
    pairs = uniform_random_heads(cfg.num_hidden_layers, cfg.num_attention_heads,
                                 ratio, seed)
    small = slice_heads(model, pairs).eval()
    if verify_batch is not None:
        diff = verify_slicing(model, small, pairs, verify_batch)
        with torch.no_grad():
            scale = model(**verify_batch).logits.abs().max().item()
        if diff / max(scale, 1e-12) > 1e-3:
            raise SlicingError(f"sliced model diverges from the masked model "
                               f"(max abs diff {diff:.3g} on logits of scale {scale:.3g})")
    return small


def nf4_available(device) -> bool:
    from sal.quantize import has_bitsandbytes
    return device.type == "cuda" and has_bitsandbytes()


def benchmark_variant(model, key, val_split, fi_batches, device, use_amp,
                      prune_seeds, skip_int4=False, verify_batch=None):
    """Accuracy of one trained variant under every compression setting.

    Pruned settings are averaged over ``prune_seeds`` head selections, and
    ``P50+I4`` quantizes each 50%-sliced model. The per-seed values are kept
    alongside the mean.
    """
    from sal import quantize
    from sal.scanner import FIScanner

    res = {}

    def cell(name, values, extra=None):
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        res[name] = {"accuracy": mean, "std": std if len(values) > 1 else None,
                     "per_seed": values, **(extra or {})}
        spread = f" ± {std * 100:.2f}" if len(values) > 1 else ""
        log(f"    {key:<8} {CONFIG_LABELS.get(name, name):<10} "
            f"{mean * 100:6.2f}%{spread}")

    cell("fp16", [accuracy(model, val_split, device, use_amp)])
    try:
        res["fp16"]["fi_score"] = FIScanner(
            model, fi_batches, num_samples=sum(b["pixel_values"].shape[0]
                                               for b in fi_batches)).scan().fi_score
    except Exception as e:  # noqa: BLE001 — FI is context, never worth losing a run over
        res["fp16"]["fi_score"] = None
        log(f"    FI scan skipped: {type(e).__name__}: {e}")

    do_nf4 = nf4_available(device) and not skip_int4
    p50_int4, p50_nf4 = [], []
    for ratio, name in ((0.33, "prune33"), (0.50, "prune50")):
        accs = []
        for i, s in enumerate(prune_seeds):
            small = sliced(model, ratio, s, verify_batch if i == 0 else None)
            accs.append(accuracy(small, val_split, device, use_amp))
            if name == "prune50" and not skip_int4:
                p50_int4.append(accuracy(quantize_weights(small, 4), val_split,
                                         device, use_amp))
                if do_nf4:
                    p50_nf4.append(accuracy(quantize(small, "int4", backend="bitsandbytes"),
                                            val_split, device, use_amp))
            del small
        cell(name, accs, {"heads_removed_per_layer":
                          int(round(ratio * model.config.num_attention_heads))})

    cell("int8", [accuracy(quantize_weights(model, 8), val_split, device, use_amp)])
    if not skip_int4:
        cell("int4", [accuracy(quantize_weights(model, 4), val_split, device, use_amp)])
        cell("prune50_int4", p50_int4)
        if do_nf4:
            cell("int4_nf4", [accuracy(quantize(model, "int4", backend="bitsandbytes"),
                                       val_split, device, use_amp)])
            cell("prune50_int4_nf4", p50_nf4)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def measure_budget(model, device, runs_cpu: int, runs_gpu: int, skip_int4: bool):
    """Size and latency per compression setting — once, since all variants share them."""
    from sal import measure_latency, quantize

    shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    s33, s50 = sliced(model, 0.33, 0), sliced(model, 0.50, 0)

    def lat(m, dev, runs):
        try:
            return measure_latency(m, shape, device=dev, warmup=max(2, runs // 5), runs=runs)
        except Exception as e:  # noqa: BLE001 — a missing number beats a lost run
            log(f"    latency on {dev} skipped: {type(e).__name__}: {e}")
            return None

    def int8_cpu(m):
        try:
            return lat(quantize(m, "int8", backend="torch_ao"), "cpu", runs_cpu)
        except Exception as e:  # noqa: BLE001
            log(f"    torch.ao INT8 latency skipped: {type(e).__name__}: {e}")
            return None

    gpu = device.type == "cuda"
    budget = {
        "fp16": {"size_mb": size_mb(model, None), "latency_cpu_ms": lat(model, "cpu", runs_cpu),
                 "latency_gpu_ms": lat(model, "cuda", runs_gpu) if gpu else None},
        "prune33": {"size_mb": size_mb(s33, None), "latency_cpu_ms": lat(s33, "cpu", runs_cpu),
                    "latency_gpu_ms": lat(s33, "cuda", runs_gpu) if gpu else None},
        "prune50": {"size_mb": size_mb(s50, None), "latency_cpu_ms": lat(s50, "cpu", runs_cpu),
                    "latency_gpu_ms": lat(s50, "cuda", runs_gpu) if gpu else None},
        "int8": {"size_mb": size_mb(model, 8), "latency_cpu_ms": int8_cpu(model),
                 "latency_gpu_ms": None},
    }
    if not skip_int4:
        budget["int4"] = {"size_mb": size_mb(model, 4), "latency_cpu_ms": None,
                          "latency_gpu_ms": None}
        budget["prune50_int4"] = {"size_mb": size_mb(s50, 4), "latency_cpu_ms": None,
                                  "latency_gpu_ms": None}
    budget["prune50_int8_cpu_reference"] = {"size_mb": size_mb(s50, 8),
                                            "latency_cpu_ms": int8_cpu(s50),
                                            "latency_gpu_ms": None}
    for k, v in budget.items():
        c, g = v["latency_cpu_ms"], v["latency_gpu_ms"]
        log(f"    {k:<28} {v['size_mb']:7.1f} MB   CPU "
            f"{'n/a' if c is None else f'{c:.1f}ms':>8}   GPU "
            f"{'n/a' if g is None else f'{g:.1f}ms':>8}")
    return budget


# --------------------------------------------------------------------- output
def _box():
    """Box-drawing glyphs, or ASCII when the terminal cannot encode them.

    The table is the last thing printed after an hour of compute. A cp1252
    console would raise UnicodeEncodeError on the box characters and lose it.
    """
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "═─".encode(enc)
        return "═", "─"
    except (UnicodeEncodeError, LookupError):
        return "=", "-"


def summary_table(results: dict, budget: dict, epochs: int, smoke: bool) -> str:
    heavy, light = _box()
    cols = [c for c in CONFIGS if any(c in results.get(v[0], {}) for v in VARIANTS)]
    w0, w = 12, 10
    lines = [f"SAL+QAT Prototype - ViT-B/16 on ImageNet-100 "
             f"({epochs} epoch{'s' if epochs != 1 else ''}{', SMOKE' if smoke else ''})",
             heavy * (w0 + w * len(cols)),
             "",
             " " * w0 + "".join(f"{CONFIG_LABELS[c]:>{w}}" for c in cols),
             " " * w0 + "".join(f"{light * (w - 2):>{w}}" for c in cols)]

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    for key, label, _, _ in VARIANTS:
        r = results.get(key)
        if r:
            lines.append(f"{label:<{w0}}" + "".join(
                f"{pct(r.get(c, {}).get('accuracy')):>{w}}" for c in cols))
    lines.append(heavy * (w0 + w * len(cols)))

    def fmt(c, field, unit):
        v = budget.get(c, {}).get(field)
        return "n/a" if v is None else f"{v:.0f}{unit}"

    lines.append(f"{'Size':<{w0}}" + "".join(f"{fmt(c, 'size_mb', 'MB'):>{w}}" for c in cols))
    lines.append(f"{'CPU latency':<{w0}}"
                 + "".join(f"{fmt(c, 'latency_cpu_ms', 'ms'):>{w}}" for c in cols))

    base = results.get("baseline", {}).get(KEY_CELL, {}).get("accuracy")
    if base is not None:
        lines += ["", f"Key cell - {CONFIG_LABELS[KEY_CELL]} accuracy, change vs baseline:"]
        deltas = {}
        for key, label, _, _ in VARIANTS[1:]:
            v = results.get(key, {}).get(KEY_CELL, {}).get("accuracy")
            if v is not None:
                deltas[key] = v - base
                lines.append(f"  {label:<10} {v * 100:6.2f}%  ({deltas[key] * 100:+.2f}pp)")
        if len(deltas) == 3:
            s, q, b = deltas["sal"], deltas["qat"], deltas["sal_qat"]
            lines.append(f"  additive prediction (SAL + QAT gains): {(s + q) * 100:+.2f}pp; "
                         f"measured SAL+QAT: {b * 100:+.2f}pp")
            best = max(s, q)
            if b > best + 0.01:
                verdict = "SAL+QAT beats both single methods by >1pp: direction supported."
            elif b > best:
                verdict = ("SAL+QAT is ahead of both single methods, by less than 1pp: "
                           "not shown on one seed.")
            else:
                verdict = "SAL+QAT does not beat the better single method: not supported."
            lines.append(f"  verdict: {verdict}")
    lines += ["",
              "Pruned cells are means over random head selections (same heads for every "
              "variant).",
              "INT8/INT4 = per-channel weight round-trip, the grid QAT trains against. "
              "Size is computed.",
              "CPU latency: INT8 is real torch.ao dynamic INT8. PyTorch has no CPU INT4 "
              "kernel, so INT4 is n/a."]
    nf4 = [k for k, *_ in VARIANTS if "prune50_int4_nf4" in results.get(k, {})]
    if nf4:
        lines.append("bitsandbytes NF4 cross-check, P50+NF4: " + ", ".join(
            f"{k} {results[k]['prune50_int4_nf4']['accuracy'] * 100:.1f}%" for k in nf4))
    if smoke:
        lines.append("SMOKE RUN - 200 images, 1 epoch. These numbers mean nothing.")
    return "\n".join(lines)


def make_figures(outdir: Path, results: dict, budget: dict) -> list:
    """accuracy_by_compression.png and size_vs_accuracy.png. Never fatal."""
    figs = outdir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not installed — skipping figures (metrics are saved).")
        return []

    # Fixed categorical order: one hue per variant, the same in both figures.
    colors = {"baseline": "#2a78d6", "sal": "#eb6834", "qat": "#1baf7a",
              "sal_qat": "#eda100"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    cols = [c for c in CONFIGS if any(c in results.get(v[0], {}) for v in VARIANTS)]
    present = [v for v in VARIANTS if v[0] in results]
    written = []

    def attempt(name, fn):
        try:
            fn()
            written.append(name)
            log(f"    {name}")
        except Exception as e:  # noqa: BLE001
            log(f"    {name} FAILED: {type(e).__name__}: {e}")
        finally:
            plt.close("all")

    def bars():
        fig, ax = plt.subplots(figsize=(10, 4.8))
        n = len(present)
        width = 0.8 / n
        for i, (key, label, _, _) in enumerate(present):
            xs = [j + (i - (n - 1) / 2) * width for j in range(len(cols))]
            ys = [results[key].get(c, {}).get("accuracy") or 0.0 for c in cols]
            ax.bar(xs, [y * 100 for y in ys], width * 0.9, label=label,
                   color=colors[key], edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(cols)), [CONFIG_LABELS[c] for c in cols])
        ax.set_ylabel("Top-1 accuracy (%)")
        lo = min((results[k].get(c, {}).get("accuracy") or 1.0)
                 for k, *_ in present for c in cols) * 100
        ax.set_ylim(max(0, lo - 5), 100)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title("Accuracy by compression setting", loc="left", color=ink)
        ax.legend(frameon=False, ncol=len(present), loc="upper right")
        fig.tight_layout()
        fig.savefig(figs / "accuracy_by_compression.png", dpi=150)

    def scatter():
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, label, _, _ in present:
            pts = [(budget[c]["size_mb"], results[key][c]["accuracy"] * 100, c)
                   for c in cols if c in budget and c in results[key]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", ms=8,
                    color=colors[key], label=label, markeredgecolor="white",
                    markeredgewidth=1.5)
        for c in cols:                # label each setting once, beside the baseline mark
            if c in budget and c in results.get("baseline", {}):
                ax.annotate(CONFIG_LABELS[c],
                            (budget[c]["size_mb"], results["baseline"][c]["accuracy"] * 100),
                            textcoords="offset points", xytext=(8, -3), color=muted,
                            fontsize=8)
        ax.set_xlabel("Deployed size (MB, computed)")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title("Size vs accuracy", loc="left", color=ink)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(figs / "size_vs_accuracy.png", dpi=150)

    attempt("accuracy_by_compression.png", bars)
    attempt("size_vs_accuracy.png", scatter)
    return written


# ----------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SAL + QAT additivity prototype on ViT-B/16 / ImageNet-100",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 200 images, INT8 only — validates the path, means nothing")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"default {DEFAULT_EPOCHS}, or 1 under --smoke (explicit value wins)")
    p.add_argument("--output", default="results/", help="output directory")
    p.add_argument("--seed", type=int, default=42, help="training seed, shared by all arms")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--mask-ratio", type=float, default=MASK_RATIO,
                   help="fraction of heads SAL prunes by the end of training")
    p.add_argument("--n-train", type=int, default=None,
                   help=f"training images (default {N_TRAIN})")
    p.add_argument("--n-val", type=int, default=None,
                   help=f"validation images (default {N_VAL}, the full val split)")
    p.add_argument("--prune-seeds", type=int, default=3,
                   help="random head selections averaged per pruning setting")
    p.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS],
                   choices=[v[0] for v in VARIANTS], help="subset of arms to run")
    p.add_argument("--manual-qat", action="store_true",
                   help="skip torch.ao prepare_qat and use the manual STE fake-quantizer")
    p.add_argument("--no-save-models", action="store_true",
                   help="do not write checkpoints to <output>/models/ (~350MB each)")
    p.add_argument("--reuse-models", action="store_true",
                   help="load <output>/models/<variant>.pt instead of retraining when present")
    p.add_argument("--workers", type=int, default=8, help="dataset decoding workers")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()

    sizes = dict(n_train=N_TRAIN, n_val=N_VAL, n_fi=N_FI, epochs=DEFAULT_EPOCHS,
                 latency_runs_cpu=LATENCY_RUNS_CPU, latency_runs_gpu=LATENCY_RUNS_GPU,
                 prune_seeds=args.prune_seeds)
    if args.smoke:
        sizes.update(SMOKE)
    for k in ("epochs", "n_train", "n_val"):
        if getattr(args, k) is not None:
            sizes[k] = getattr(args, k)
    epochs = sizes["epochs"]
    skip_int4 = args.smoke
    prune_seeds = [args.seed + i for i in range(sizes["prune_seeds"])]

    outdir = Path(args.output)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    models_dir = outdir / "models"

    rule("Environment")
    device, vram = detect_device()
    use_amp = device.type == "cuda"
    log(f"  CPU threads: {torch.get_num_threads()}  (set OMP_NUM_THREADS on containers)")
    log(f"  output: {outdir.resolve()}")
    log(f"  mode:   {'SMOKE' if args.smoke else 'full'} | epochs={epochs} | "
        f"train={sizes['n_train']} val={sizes['n_val']} | seed={args.seed} | "
        f"mask_ratio={args.mask_ratio} | QAT {QAT_BITS}-bit | "
        f"prune seeds {prune_seeds}{' | INT4 skipped' if skip_int4 else ''}")

    rule("Data")
    log(f"  model:   {MODEL_ID}")
    log(f"  dataset: {DATASET_ID}")
    train_split, val_split, num_classes = build_data(sizes, args.seed, args.workers)
    log(f"  {train_split[0].shape[0]} train / {val_split[0].shape[0]} val images, "
        f"{num_classes} classes")
    n_fi = min(sizes["n_fi"], val_split[0].shape[0])
    fi_x = normalize(val_split[0][:n_fi]).to(device)
    fi_batches = [{"pixel_values": fi_x[i:i + 16]} for i in range(0, n_fi, 16)]
    verify_batch = {"pixel_values": fi_x[:4]}

    results, training, budget = {}, {}, {}
    for key, label, use_sal, use_qat in VARIANTS:
        if key not in args.variants:
            continue
        rule(f"Variant: {label}  (SAL={'on' if use_sal else 'off'}, "
             f"QAT={'on' if use_qat else 'off'})")
        ckpt = models_dir / f"{key}.pt"
        if args.reuse_models and ckpt.exists():
            set_seed(args.seed)
            model = load_model(num_classes)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.to(device).eval()
            training[key] = {"reused_checkpoint": str(ckpt)}
            log(f"  reused {ckpt}")
        else:
            model, training[key] = train_variant(
                key, use_sal, use_qat, num_classes, train_split, epochs,
                args.batch_size, args.seed, device, use_amp, args.mask_ratio,
                force_manual_qat=args.manual_qat)
            if not args.no_save_models:
                models_dir.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           ckpt)
                log(f"  checkpoint -> {ckpt}")

        log("  benchmarking:")
        results[key] = benchmark_variant(model, key, val_split, fi_batches, device,
                                         use_amp, prune_seeds, skip_int4, verify_batch)
        if not budget:
            log("  size / latency (shared by every variant):")
            budget = measure_budget(model, device, sizes["latency_runs_cpu"],
                                    sizes["latency_runs_gpu"], skip_int4)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ save
    rule("Saving")
    table = summary_table(results, budget, epochs, args.smoke)
    (outdir / "summary_table.txt").write_text(table + "\n", encoding="utf-8")
    figures = make_figures(outdir, results, budget)
    payload = {
        "model": MODEL_ID, "dataset": DATASET_ID, "smoke": args.smoke,
        "seed": args.seed, "epochs": epochs, "batch_size": args.batch_size,
        "n_train": train_split[0].shape[0], "n_val": val_split[0].shape[0],
        "num_classes": num_classes, "mask_ratio": args.mask_ratio,
        "qat_bits": QAT_BITS, "prune_seeds": prune_seeds, "device": str(device),
        "vram_gb": round(vram, 1),
        "lr": {"backbone": LR_BACKBONE, "head": LR_HEAD, "weight_decay": WEIGHT_DECAY,
               "warmup_fraction": WARMUP_FRACTION, "schedule": "linear warmup + cosine"},
        "caveats": {
            "quantization": "int8/int4 accuracy = per-output-channel symmetric weight "
                            "round-trip on every nn.Linear except the classifier; the "
                            "same grid QAT trains against. *_nf4 = real bitsandbytes NF4.",
            "size": "computed: fp16 non-quantized params, bits/8 bytes per quantized "
                    "weight + one fp16 scale per output channel",
            "latency": "measured once per setting (architecture-dependent only). int8 "
                       "CPU latency is torch.ao dynamic INT8; no CPU INT4 kernel exists "
                       "in stock PyTorch, so int4 latency is null.",
            "pruning": "uniform random heads per layer, sliced with slice_heads(); mean "
                       "over prune_seeds; identical selections for every variant",
            "fi": "fi_score on the dense trained model only",
            "seed": "one training seed; treat sub-1pp gaps as not shown",
        },
        "training": training, "results": results, "budget": budget,
        "figures": figures, "wall_clock_s": round(time.time() - started, 1),
    }
    (outdir / "sal_qat_prototype.json").write_text(json.dumps(payload, indent=2))
    log(f"  metrics -> {outdir / 'sal_qat_prototype.json'}")
    log(f"  table   -> {outdir / 'summary_table.txt'}")

    rule("Summary")
    log(table)
    log()
    log(f"wall clock: {(time.time() - started) / 60:.1f} min")
    log(f"Everything is in {outdir.resolve()} — scp that directory back.")
    return payload


if __name__ == "__main__":
    main()
