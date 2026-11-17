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
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

MODEL_ID = "google/vit-base-patch16-224"
DATASET_ID = "clane9/imagenet-100"
# Printed wherever a result names its data. ImageNet-1k here is the ungated
# mirror: ILSVRC/imagenet-1k is gated and needs an accepted-terms token.
# The class count each dataset must load. build_data refuses a mismatch.
DATASET_CLASSES = {
    "clane9/imagenet-100": 100,
    "evanarlian/imagenet_1k_resized_256": 1000,
}
DATASET_LABELS = {
    "clane9/imagenet-100": "ImageNet-100",
    "evanarlian/imagenet_1k_resized_256":
        "ImageNet-1k (resized 256px, evanarlian/imagenet_1k_resized_256)",
}


# DataLoader workers for the training and evaluation loaders. 0 keeps the
# in-RAM behaviour every earlier experiment ran with; main() raises it when the
# split is memory-mapped, where the reads are the thing to overlap.
LOADER_WORKERS = 0
# Block size for _BlockShuffle, 0 for a full shuffle. Also set by main() only
# for a memory-mapped split.
LOADER_BLOCK = 0


def dataset_label(dataset_id: str | None) -> str:
    """How a result file should name this dataset."""
    did = dataset_id or DATASET_ID
    return DATASET_LABELS.get(did, did)
IMAGE_SIZE = 224

VARIANTS = (                      # (key, label, use_sal, use_qat)
    ("baseline", "Baseline", False, False),
    ("sal", "SAL only", True, False),
    ("qat", "QAT only", False, True),
    ("sal_qat", "SAL + QAT", True, True),
)

# Control arms, opt-in via --variants. They exist to answer "is SAL's gain
# actually SAL's?", and are deliberately NOT in the default set, so the four
# validated arms above keep running exactly as before.
#
#   drophead  — classic stochastic head dropout: ``mask_ratio`` of heads masked
#               every step, resampled every step, on from step 0, never held.
#               If this matches SAL, the gain is ordinary regularisation and
#               nothing about progressive structural damage is needed.
#   nonaccum  — SAL's schedule with the accumulation removed: the pruned *count*
#               ramps over the same window, but the identities are resampled at
#               every step so no head stays dead. Isolates accumulation alone,
#               holding the ramp fixed. CLAUDE.md calls accumulation "the causal
#               mechanism"; this is the arm that tests it.
#   progressive — DropHead with a ramp. The dropped *count* grows linearly from
#               PROGRESSIVE_START_RATIO to mask_ratio across the whole run, and
#               identities are resampled every step. Tests whether the ramp in
#               SAL's schedule is worth anything once accumulation is gone.
#   fi_drophead — DropHead that biases its draw towards structurally fragile
#               heads, read off the model's own attention graph and refreshed
#               during training. Same expected drop rate as plain DropHead, so
#               only *which* heads get dropped differs. See `head_fragility`.
CONTROL_VARIANTS = (              # (key, label, mask_mode, use_qat)
    ("drophead", "DropHead ctrl", "drophead", False),
    ("nonaccum", "No-accum ctrl", "nonaccum", False),
    ("drophead_qat", "DropHead + QAT", "drophead", True),
    ("nonaccum_qat", "No-accum + QAT", "nonaccum", True),
    ("progressive", "Progressive DH", "progressive", False),
    ("fi_drophead", "FI-guided DH", "fi_drophead", False),
    ("progressive_qat", "Progressive + QAT", "progressive", True),
    ("fi_drophead_qat", "FI-guided + QAT", "fi_drophead", True),
)

PROGRESSIVE_START_RATIO = 0.1     # where the progressive ramp begins
FI_FRAGILE_BOOST = 2.0            # drop weight for fragile heads, before renormalising
FI_RESILIENT_BOOST = 0.5          # drop weight for the rest
FI_RESCAN_EPOCHS = 2              # refresh the fragility read every N epochs

# One list the main loop walks, with use_sal normalised into a mask mode.
ALL_VARIANTS = tuple(
    (k, label, "sal" if use_sal else "none", use_qat)
    for k, label, use_sal, use_qat in VARIANTS
) + CONTROL_VARIANTS
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


def validate_probe(probe, where: str = "probe") -> dict:
    """Check a probe is what FI assumes it is, and say so loudly if not.

    FI is a function of the **(model, probe)** pair, so a malformed probe does
    not raise anywhere -- it silently returns a different number. Two failures
    in this campaign were exactly that:

    * **Unnormalized input.** exp37 and exp42 built their probes from the raw
      uint8 cache without passing it through :func:`normalize`. Both runs
      completed, wrote JSON and produced a table; the numbers were of a
      different quantity and two full GPU runs had to be redone. Nothing in
      the stack objected, because ``model(pixel_values=...)`` accepts any
      float tensor of the right shape.
    * **An unrecorded probe size.** The same checkpoint and the same
      normalized construction read FI 0.151 at 64 images and 0.071-0.082 at
      500 (``results/lumi_v060/EXP42_FI_THRESHOLDS.md``). Ragged batches make
      the recorded ``num_samples`` a fiction.

    Returns the probe's statistics so a caller can record them. Raises
    ``ValueError`` on anything it cannot vouch for.
    """
    if not isinstance(probe, (list, tuple)) or not probe:
        raise ValueError(
            "{} must be a non-empty list of batches, got {!r}".format(
                where, type(probe).__name__))
    sizes, lo, hi, total = set(), None, None, 0
    for i, batch in enumerate(probe):
        if not isinstance(batch, dict) or "pixel_values" not in batch:
            raise ValueError(
                "{}[{}] must be a dict with a 'pixel_values' key, got "
                "{!r}".format(where, i, batch))
        x = batch["pixel_values"]
        if not torch.is_floating_point(x):
            raise ValueError(
                "{}[{}]['pixel_values'] has dtype {}, not a float type. An "
                "integer probe is the unnormalized cache: pass it through "
                "normalize() or build it with build_probe(). FI computed "
                "from uint8 input is a different quantity and nothing "
                "downstream will tell you.".format(where, i, x.dtype))
        if not bool(torch.isfinite(x).all()):
            raise ValueError(
                "{}[{}]['pixel_values'] contains non-finite values".format(
                    where, i))
        sizes.add(int(x.shape[0]))
        total += int(x.shape[0])
        bl, bh = float(x.min()), float(x.max())
        lo = bl if lo is None else min(lo, bl)
        hi = bh if hi is None else max(hi, bh)
    # A mean/std-normalized image tensor always straddles zero; uint8/255 does
    # not. This is the check that would have caught exp37 and exp42 on their
    # first gate instead of after two full runs.
    if lo >= -0.01:
        raise ValueError(
            "{} looks unnormalized: values span [{:.3f}, {:.3f}] and a "
            "mean/std-normalized probe straddles zero. Raw uint8/255 input "
            "lands in [0, 1] and raw bytes in [0, 255]; either produces a "
            "valid-looking FI that measures something else. Build the probe "
            "with build_probe().".format(where, lo, hi))
    if len(sizes) > 1:
        raise ValueError(
            "{} has ragged batch sizes {}. num_samples and batch_size are "
            "recorded on every FI figure and a ragged probe makes both "
            "meaningless.".format(where, sorted(sizes)))
    return {"batches": len(probe), "batch_size": sizes.pop(),
            "images": total, "min": lo, "max": hi}


def build_probe(images_uint8, n_batches: int, batch_size: int, device=None,
                where: str = "probe"):
    """The one probe builder. Every experiment calls this, none rolls its own.

    Rule 1 of audit 2. Before this existed there were six independent probe
    constructions across the three repositories -- two of them unnormalized,
    one of them normalizing conditionally on a flag -- and FI numbers from
    different scripts were not comparable without reading each script. The
    cross-harness check (exp45) only proved the *scanner* agrees; what
    disagreed was what each script fed it.

    ``images_uint8`` is a uint8 ``[N, 3, H, W]`` block from the cache. The
    first ``n_batches * batch_size`` images are taken in order, so the caller
    controls the draw by slicing the block it passes, and two callers passing
    the same block get the same probe.
    """
    need = n_batches * batch_size
    have = int(images_uint8.shape[0])
    if have < need:
        raise ValueError(
            "{} needs {} images ({} batches of {}), got {}".format(
                where, need, n_batches, batch_size, have))
    probe = [{"pixel_values": normalize(
        images_uint8[i * batch_size:(i + 1) * batch_size]).to(device)
        if device is not None else normalize(
            images_uint8[i * batch_size:(i + 1) * batch_size])}
        for i in range(n_batches)]
    validate_probe(probe, where)
    return probe


# Values a new script must reproduce before it is allowed to report new ones.
# Rule 2 of audit 2. Every entry is published, with its source, and every one
# of them is a (model, probe, protocol) triple -- not a property of the model
# alone. Quote the source, not this dict.
REFERENCE_VALUES = {
    "vit_b16_in1k_pretrained_top1": {
        "value": 0.7844, "tol": 0.0005,
        "source": "results/lumi_v060/PHASE3_IMAGENET1K.md",
        "note": "ViT-B/16 on the resized-256 mirror, full 50k val split"},
    "vit_b16_in1k_pretrained_int4_top1": {
        "value": 0.77228, "tol": 0.0005,
        "source": "results/lumi_v060/PHASE3_IMAGENET1K.md",
        "note": "fp32 rounding, scored under fp16 autocast"},
    "vit_b16_in1k_control_2ep_top1": {
        "value": 0.81554, "tol": 0.0015,
        "source": "results/lumi_v060/CONTROL_CONVENTION.md",
        "note": "3 seeds, nothing pruned; the matched control"},
    # A band, not a point, and deliberately a generous one.
    #
    # The dense FI of this checkpoint has been measured on eleven independent
    # probe draws across three scripts, and it spans **0.064078 to 0.082524**
    # (exp37 0.080583 / 0.069903 / 0.065049 and 0.067961 / 0.066019 /
    # 0.074757; exp42 0.079612 / 0.070874 / 0.081553 and 0.078641 / 0.064078;
    # exp47 0.066990 / 0.070874 / 0.082524). That spread is not error -- it is
    # the probe-noise floor, and it is the measurement
    # (EXP42_FI_THRESHOLDS.md). So a script that draws its probe at random
    # cannot be gated on a point.
    #
    # It cannot be gated on the measured min/max either. A min/max over
    # eleven draws is not a tolerance interval: the twelfth draw exceeds it
    # about as often as not, and a gate that fires on a legitimate reading one
    # edge outside the previous extreme teaches everyone to pass
    # --no-reference-gate. That happened on the first run of this gate, where
    # 0.064078 (a real reading, and one exp30 also recorded) sat 0.001 under a
    # band built from the earlier draws.
    #
    # So the band is set to what the gate is *for*: catching a probe that is
    # wrong by an order of magnitude, which is the failure this campaign
    # actually had. The unnormalized uint8 run read **0.132039** (136 edges,
    # roughly double) and a 64-image probe reads **0.151456**. Both are far
    # outside 0.05-0.10, while every normalized 512-image draw sits well
    # inside it with ~15 edges of margin at the bottom and ~18 at the top.
    "vit_b16_dense_fi_band": {
        "lo": 0.050, "hi": 0.100,
        "source": "results/lumi_v060/EXP42_FI_THRESHOLDS.md",
        "note": "11 normalized draws span 0.064078-0.082524; the band is "
                "wider on purpose. It catches an unnormalized probe (0.132) "
                "or the wrong probe size (0.151), not sampling noise. A "
                "reading outside it means the probe is wrong, not that the "
                "model is interesting."},
    "vit_b16_dense_fi_500img": {
        "value": 0.070874, "tol": 0.0,
        "source": "results/lumi_v060/EXP42_FI_THRESHOLDS.md",
        "note": "first 512 images of the seed-42 val shuffle, 32 batches of "
                "16, num_samples=500, edge density 0.10. Exact: the scan is "
                "deterministic on a fixed probe (exp45). A different probe "
                "reads 0.067-0.083 -- that spread is the floor, not an "
                "error bar, so this value is only a reference gate for THIS "
                "probe."},
}


def check_reference(key: str, measured: float, log_fn=None) -> float:
    """Fail the run if a known value does not reproduce.

    Rule 2 of audit 2: a script reproduces something already published before
    it reports anything new. The campaign's two silent-wrong-number failures
    (an unnormalized probe, a 5-epoch arm scored against a 2-epoch control)
    would both have been caught in the gate by one line of this.
    """
    if key not in REFERENCE_VALUES:
        raise KeyError(
            "no reference value named {!r}. Known: {}".format(
                key, sorted(REFERENCE_VALUES)))
    ref = REFERENCE_VALUES[key]
    if "lo" in ref:
        ok = ref["lo"] <= measured <= ref["hi"]
        expected = "[{:.6f}, {:.6f}]".format(ref["lo"], ref["hi"])
        delta = 0.0 if ok else min(abs(measured - ref["lo"]),
                                   abs(measured - ref["hi"]))
    else:
        delta = abs(measured - ref["value"])
        ok = delta <= ref["tol"]
        expected = "{:.6f} +/- {:.6f}".format(ref["value"], ref["tol"])
    (log_fn or log)("  reference {}: measured {:.6f} vs {} ({}) -> {}".format(
        key, measured, expected, ref["source"],
        "OK" if ok else "MISMATCH"))
    if not ok:
        raise RuntimeError(
            "reference gate {} failed: measured {:.6f}, expected {} (off by "
            "{:.6f}). Source: {}. {}\nA new number from this run would not "
            "be comparable to anything already published, so the run stops "
            "here.".format(key, measured, expected, delta, ref["source"],
                           ref["note"]))
    return delta


def _split_names(ds) -> tuple[str, str]:
    """``(train_key, val_key)`` for this dataset.

    ImageNet-100 calls the held-out split ``validation``; the ImageNet-1k mirror
    calls it ``val`` and also carries an unlabelled ``test``. Guessing wrong
    here silently evaluates on the wrong data, so the keys are resolved
    explicitly and the error names what was available.
    """
    train = "train" if "train" in ds else None
    val = next((k for k in ("validation", "val", "valid") if k in ds), None)
    if train is None or val is None:
        raise SystemExit(f"dataset has splits {sorted(ds)}; need a train split and "
                         "one of validation/val/valid")
    return train, val


def build_data(sizes: dict, seed: int, workers: int = 8,
               dataset_id: str | None = None, cache_dir=None):
    """Dataset subsets as uint8 tensors, in memory or memory-mapped.

    Images are resized to 224x224 once, up front, and kept as uint8 (147KB each),
    so every arm trains on byte-identical inputs and no epoch waits on JPEG
    decoding. Returns ``(train, val, num_classes)`` with each split an
    ``(images_uint8, labels)`` pair.

    ``cache_dir`` writes each split to a ``.npy`` there and memory-maps it back.
    The full ImageNet-1k train split is 193 GB as uint8, which does not fit in a
    small-g job's 60 GB, and re-decoding 1.28M JPEGs once per epoch per arm
    would put JPEG decoding on the critical path of every measurement. An
    existing cache file of the right shape is reused, so the arms of a sweep
    decode the data once between them.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    ds = load_dataset(dataset_id or DATASET_ID)
    train_key, val_key = _split_names(ds)
    log(f"  dataset: {dataset_id or DATASET_ID} | "
        f"{len(ds[train_key])} train / {len(ds[val_key])} {val_key} available")

    def collate(rows):
        imgs = [r["image"].convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), 2)  # bilinear
                for r in rows]
        x = torch.stack([torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
                         .view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1) for im in imgs])
        return x, torch.tensor([r["label"] for r in rows])

    def encode(split, n, label):
        n = min(n, len(split))
        # Always shuffle, including for a whole split. Skipping it leaves the
        # cache in dataset order -- sorted by class -- which forces the training
        # loader to read 193 GB randomly to get a mixed batch: measured 104 img/s
        # against 376 on a subset that fit in the page cache. Shuffling here
        # costs nothing that matters (the arrow source is 24 GB and cacheable;
        # JPEG decoding dominates and is order-independent) and lets training
        # read the cache in contiguous blocks.
        subset = split.shuffle(seed=seed)
        if n != len(split):
            subset = subset.select(range(n))
        dl = DataLoader(subset, batch_size=64, num_workers=workers, collate_fn=collate)
        prog = Progress(len(dl), f"encoding {label}")
        if cache_dir is None:
            xs, ys = [], []
            for x, y in dl:
                xs.append(x)
                ys.append(y)
                prog.step()
            prog.done()
            return torch.cat(xs), torch.cat(ys)

        import numpy as np

        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        # "-shuf" is a cache-format marker, not decoration: a file written by
        # the earlier encoder is in class order and must not be reused.
        tag = f"{(dataset_id or DATASET_ID).replace('/', '__')}-{label}-{n}-s{seed}-shuf"
        xpath, ypath = cache / f"{tag}-x.npy", cache / f"{tag}-y.npy"
        if xpath.exists() and ypath.exists():
            mm = np.load(xpath, mmap_mode="r")
            if mm.shape == (n, 3, IMAGE_SIZE, IMAGE_SIZE):
                prog.done()
                log(f"  reusing cache {xpath.name} ({mm.nbytes / 1e9:.1f} GB)")
                return torch.from_numpy(mm), torch.from_numpy(np.load(ypath))
            log(f"  cache {xpath.name} has shape {mm.shape}, expected "
                f"{(n, 3, IMAGE_SIZE, IMAGE_SIZE)}; re-encoding")
            del mm
        # The rename protects against interruption; the lock protects
        # against concurrency. Without it, several jobs starting together
        # for a new seed all encode the same file, share one .part buffer,
        # and all but one die on FileNotFoundError when the winner renames
        # it. That is what wave 3 lost most of its new-seed jobs to.
        want_shape = (n, 3, IMAGE_SIZE, IMAGE_SIZE)
        lock = _claim_cache_lock(xpath)
        if lock is None:
            got = _wait_for_cache(xpath, ypath, n, want_shape)
            if got is not None:
                prog.done()
                return got
            lock = _claim_cache_lock(xpath)

        # Per process, so two writers can never share one buffer even if a
        # lock is ever taken over wrongly.
        part = xpath.with_suffix(".part.%s.npy"
                                 % os.environ.get("SLURM_JOB_ID",
                                                  str(os.getpid())))
        out = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8,
                                        shape=(n, 3, IMAGE_SIZE, IMAGE_SIZE))
        ys, i = [], 0
        for x, y in dl:
            out[i:i + x.shape[0]] = x.numpy()
            i += x.shape[0]
            ys.append(y)
            prog.step()
        prog.done()
        out.flush()
        del out
        labels = torch.cat(ys)
        np.save(ypath, labels.numpy())
        part.replace(xpath)
        if lock is not None:
            try:
                lock.unlink()
            except OSError:
                pass
        mm = np.load(xpath, mmap_mode="r")
        log(f"  cached {xpath.name} ({mm.nbytes / 1e9:.1f} GB)")
        return torch.from_numpy(mm), labels

    train = encode(ds[train_key], sizes["n_train"], "train")
    val = encode(ds[val_key], sizes["n_val"], "val")
    num_classes = len(ds[train_key].features["label"].names)
    # Audit 2, rule 1: fail loudly rather than train a classifier with the
    # wrong head width. A dataset that silently loads a different class count
    # -- a changed mirror, a wrong --dataset, a cached split from another run
    # -- produces an accuracy that is not comparable to anything published,
    # and nothing downstream objects. exp29, exp31 and exp33 each read a split
    # without this check; the assertion belongs here, in the one builder, not
    # in each of them.
    expected = DATASET_CLASSES.get(dataset_id or DATASET_ID)
    if expected is not None and num_classes != expected:
        raise RuntimeError(
            "{} loaded {} classes, expected {}. Every accuracy in this "
            "project is on a fixed class count; a different one is not a "
            "variant of the same measurement. Check --dataset and the "
            "cache.".format(dataset_id or DATASET_ID, num_classes, expected))
    return train, val, num_classes


class _BlockShuffle(torch.utils.data.Sampler):
    """Shuffle in contiguous blocks, so a memory-mapped split reads sequentially.

    Indices are cut into blocks of ``block`` consecutive rows; the block order is
    shuffled and so is the order within each block. A full shuffle gives perfect
    mixing and the worst possible access pattern on a 193 GB file; this gives
    sequential runs of ``block`` rows, which is what Lustre is fast at.

    It is only sound because the cache is **written in random order** -- see the
    encoder. On a class-sorted file, blocks would be single-class batches.

    The generator is created once and advances across epochs, so successive
    passes differ, exactly as ``DataLoader(shuffle=True, generator=g)`` does.
    """

    def __init__(self, n: int, block: int, seed: int = 0):
        self.n, self.block = int(n), max(1, int(block))
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        starts = torch.arange(0, self.n, self.block)
        for b in starts[torch.randperm(starts.numel(), generator=self.g)].tolist():
            end = min(b + self.block, self.n)
            inner = torch.randperm(end - b, generator=self.g) + b
            yield from inner.tolist()


def batches(split, batch_size: int, shuffle: bool = False, seed: int = 0,
            workers: int = 0, block: int = 0):
    """A DataLoader over a ``(uint8 images, labels)`` split.

    Shuffling uses its own seeded generator, so every arm sees the same order.

    ``workers`` is 0 by default, which is right for a split held in RAM:
    indexing a resident tensor is free and a worker process would only add
    pickling. It matters for a memory-mapped split, where each row is a 147 KB
    read from Lustre, and workers let those reads overlap the compute.

    ``block`` > 0 replaces the full shuffle with :class:`_BlockShuffle`, which
    is what makes a memory-mapped split affordable: a full shuffle of a 193 GB
    file measured 3.6x slower than the same data cached. It requires a cache
    written in random order, which the encoder now does. ``block`` is ignored
    when ``shuffle`` is False, since an unshuffled pass is already sequential.
    """
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(*split)
    kw = {}
    if workers:
        # persistent_workers only when shuffling, i.e. for training, where
        # the loader is iterated once per epoch and keeping the workers
        # alive is the point. An evaluation pass is iterated ONCE, so
        # persistence buys nothing and costs the workers' lifetime: they
        # and their POSIX semaphores survive until the loader is
        # collected, and over a long job the semaphores accumulate.
        # exp33 seed 8888 died that way after 4h16 -- the recovery had
        # finished and `score()` could no longer create a DataLoader
        # ("SemLock ... FileNotFoundError"), so the cell's recovered
        # accuracy was never computed. Non-persistent workers shut down
        # when the iterator is exhausted, which is exactly what an
        # evaluation pass wants.
        kw = {"num_workers": workers, "persistent_workers": bool(shuffle),
              "prefetch_factor": 4}
    if shuffle and block:
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=_BlockShuffle(len(dataset), block, seed), **kw)
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, generator=g, **kw)


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


class ResampleMasker:
    """``HeadMasker`` with the accumulation taken out — the control arms.

    Wraps a real :class:`sal.HeadMasker` so the masking *mechanism* (the
    pre-hook on the attention output projection, the per-head slicing) is
    byte-for-byte the one SAL uses. Only ``step()`` differs: instead of
    accumulating dead heads, it clears the masks and draws a fresh random set
    every step, so no head is ever permanently removed.

    Three modes:

    ``"drophead"``    classic DropHead — the full ``mask_ratio`` from step 0.
    ``"nonaccum"``    SAL's own scheduled count, so the dropped *quantity*
                      follows the identical curve and the only thing that
                      changed is whether the same heads stay dead.
    ``"progressive"`` a linear ramp of its own, from
                      ``PROGRESSIVE_START_RATIO`` to ``mask_ratio`` across the
                      whole run rather than inside SAL's window.

    This reaches into a few private members of HeadMasker (``_masks``,
    ``_active``, ``_scheduled_count``, ``_deactivate_random_heads``). That is
    deliberate: reimplementing them would make the control differ from the
    treatment in exactly the way the control is meant to rule out.
    """

    def __init__(self, model, config, seed, mode: str = "drophead"):
        from sal import HeadMasker
        if mode not in ("drophead", "nonaccum", "progressive"):
            raise ValueError(f"unknown resample mode {mode!r}")
        self.inner = HeadMasker(model, config, seed=seed)
        self.mode = mode

    def install(self):
        self.inner.install()

    def remove(self):
        self.inner.remove()

    def apply_mask(self):
        self.inner.apply_mask()

    def remove_mask(self):
        self.inner.remove_mask()

    def unmasked(self):
        return self.inner.unmasked()

    def target_count(self, global_step: int, total_steps: int) -> int:
        inner = self.inner
        if self.mode == "nonaccum":
            if global_step / max(total_steps, 1) < inner.config.prune_start_ratio:
                return -1                               # before the window: nothing
            return inner._scheduled_count(global_step, total_steps)
        if self.mode == "progressive":
            progress = global_step / max(total_steps, 1)
            ratio = (PROGRESSIVE_START_RATIO
                     + (inner.config.prune_fraction - PROGRESSIVE_START_RATIO) * progress)
            return int(round(inner.config.total_heads * ratio))
        return inner.config.num_heads_to_prune

    def step(self, global_step: int, total_steps: int):
        inner = self.inner
        target = self.target_count(global_step, total_steps)
        if target < 0:
            inner.deactivate()
            return
        for m in inner._masks.values():                 # drop the previous set
            m.fill_(1.0)
        inner._active = True
        if target > 0:
            inner._deactivate_random_heads(target)

    @property
    def stats(self):
        return self.inner.stats


def head_fragility(model, fi_batches) -> "list[float]":
    """Per-head fragility, in [0, 1], read off the model's own attention graph.

    ``sal.compute_fi`` reports one number for the whole graph: the share of
    edges that no triangle supports, i.e. edges whose two heads share no common
    neighbour and so have no redundant pathway. This is the same quantity read
    per head instead of globally — the share of *a head's own* edges that
    nothing supports. A head whose edges are all unsupported has no backup for
    anything it carries; a head sitting in a densely triangulated
    neighbourhood does.

    An isolated head (no edges at all) scores 1.0: nothing supports it either.
    """
    import numpy as np
    from sal.fi import extract_activation_graph

    n_probe = sum(b["pixel_values"].shape[0] for b in fi_batches)
    adj = extract_activation_graph(model, fi_batches, num_samples=n_probe,
                                   batch_size=fi_batches[0]["pixel_values"].shape[0])
    A = (np.asarray(adj) != 0).astype(np.int64)
    np.fill_diagonal(A, 0)
    support = A @ A                                     # shared neighbours per pair
    out = []
    for i in range(A.shape[0]):
        nbrs = np.nonzero(A[i])[0]
        if len(nbrs) == 0:
            out.append(1.0)
            continue
        out.append(float((support[i, nbrs] == 0).sum()) / len(nbrs))
    return out


class FIDropHeadMasker(ResampleMasker):
    """DropHead that biases its draw towards structurally fragile heads.

    Same mechanism and the **same expected number of dropped heads** as plain
    DropHead — only the distribution over *which* heads changes. Fragile heads
    (fragility above the median) are dropped with weight
    ``FI_FRAGILE_BOOST``, the rest with ``FI_RESILIENT_BOOST``, and both are
    renormalised so the expected rate equals ``mask_ratio`` exactly:

        E[rate] = f * c * BOOST_f + (1 - f) * c * BOOST_r = mask_ratio
        => c = 1 / (f * BOOST_f + (1 - f) * BOOST_r)

    with ``f`` the fragile fraction. At the median split f = 0.5, giving
    per-step drop probabilities of 1.6x and 0.4x the base ratio.

    Draws are **independent Bernoulli per head**, which makes the stated
    probabilities exact but lets the per-step count fluctuate around the
    target, where plain DropHead holds it fixed. That is a real difference
    between the two arms and is the reason the count is logged.

    The graph is re-read every ``rescan_steps`` steps, with masking suspended
    for the scan: the graph must describe the model, not the model minus
    whatever was dropped this step.
    """

    def __init__(self, model, config, seed, fi_batches, rescan_steps: int):
        super().__init__(model, config, seed, mode="drophead")
        self.fi_batches = fi_batches
        self.rescan_steps = max(1, rescan_steps)
        self.model = model
        self._probs = None
        self._rescans = 0
        self._fragile_fraction = None
        self._counts = []

    def _rescan(self):
        """Refresh per-head drop probabilities from the current graph."""
        import numpy as np

        was_training = self.model.training
        with self.inner.unmasked():                     # scan the whole model
            self.model.eval()
            try:
                frag = head_fragility(self.model, self.fi_batches)
            finally:
                if was_training:
                    self.model.train()

        frag = np.asarray(frag, dtype=float)
        median = float(np.median(frag))
        fragile = frag > median
        f = float(fragile.mean())
        base = self.inner.config.prune_fraction
        c = 1.0 / (f * FI_FRAGILE_BOOST + (1.0 - f) * FI_RESILIENT_BOOST)
        p = np.where(fragile, c * FI_FRAGILE_BOOST, c * FI_RESILIENT_BOOST) * base
        self._probs = np.clip(p, 0.0, 1.0)
        self._fragile_fraction = f
        self._rescans += 1

    def step(self, global_step: int, total_steps: int):
        import numpy as np

        if self._probs is None or global_step % self.rescan_steps == 0:
            self._rescan()

        inner = self.inner
        nh = inner.config.num_heads_per_layer
        draw = np.random.default_rng(
            abs(hash((global_step, id(self)))) % (2 ** 32)).random(len(self._probs))
        drop = np.nonzero(draw < self._probs)[0]
        for m in inner._masks.values():
            m.fill_(1.0)
        inner._active = True
        for flat in drop.tolist():
            inner._masks[flat // nh][flat % nh] = 0.0
        self._counts.append(int(len(drop)))

    @property
    def stats(self):
        s = dict(self.inner.stats)
        s["fi_rescans"] = self._rescans
        s["fi_fragile_fraction"] = (round(self._fragile_fraction, 4)
                                    if self._fragile_fraction is not None else None)
        if self._counts:
            s["mean_dropped_per_step"] = round(sum(self._counts) / len(self._counts), 1)
        return s


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


def train_variant(key, mask_mode, use_qat, num_classes, train_split, epochs, batch_size,
                  seed, device, use_amp, mask_ratio, force_manual_qat=False,
                  fi_batches=None, uniform_per_layer=False):
    """Build and train one variant. Returns ``(model, info)``, with QAT stripped().

    ``mask_mode`` is ``"none"`` (no masking at all), ``"sal"`` (the validated
    accumulating schedule, via SALTrainer), or one of the controls
    ``"drophead"`` / ``"nonaccum"`` / ``"progressive"`` (see
    :class:`ResampleMasker`) or ``"fi_drophead"`` (see
    :class:`FIDropHeadMasker`, which needs ``fi_batches``).

    ``uniform_per_layer`` constrains every draw to an equal quota per layer, so
    the trained set is expressible as a ``slice_heads()`` argument. Off by
    default; the shipped behaviour is the global draw.
    """
    from sal import SALConfig, SALTrainer

    set_seed(seed)                     # identical head init across arms
    model = load_model(num_classes)
    qat_backend = None
    if use_qat:
        model, qat_backend = setup_qat(model, QAT_BITS, force_manual=force_manual_qat)
    model.to(device).train()

    loader = batches(train_split, batch_size, shuffle=True, seed=seed,
                     workers=LOADER_WORKERS, block=LOADER_BLOCK)
    total = len(loader) * epochs
    optimizer, scheduler = make_optimizer(model, total)
    prog = Progress(total, f"{key} training")
    step = make_step(device, use_amp, prog)

    t0 = time.time()
    if mask_mode == "sal":
        cfg = SALConfig.auto(model, prune_fraction=mask_ratio,
                             uniform_per_layer=uniform_per_layer)
        trainer = SALTrainer(model, cfg, optimizer, loader, scheduler=scheduler,
                             seed=seed, train_step=step)
        history = trainer.train(num_epochs=epochs)
        losses, masker_stats = history["losses"], history["masker_stats"]
        # Which heads this arm was trained without. Needed to score a model on
        # its own pruned set rather than an unrelated random one (exp12).
        pruned_heads = history.get("pruned_heads")
    elif mask_mode in ("drophead", "nonaccum", "progressive", "fi_drophead"):
        # Same hook, same schedule shape, no accumulation. Driven by hand rather
        # than SALTrainer, which owns the accumulating schedule we are removing.
        cfg = SALConfig.auto(model, prune_fraction=mask_ratio,
                             uniform_per_layer=uniform_per_layer)
        if mask_mode == "fi_drophead":
            if not fi_batches:
                raise ValueError("fi_drophead needs fi_batches to read the graph from")
            steps_per_epoch = max(1, len(loader))
            masker = FIDropHeadMasker(model, cfg, seed=seed, fi_batches=fi_batches,
                                      rescan_steps=steps_per_epoch * FI_RESCAN_EPOCHS)
        else:
            masker = ResampleMasker(model, cfg, seed=seed, mode=mask_mode)
        masker.install()
        losses, gstep = [], 0
        try:
            for _ in range(epochs):
                total_loss = 0.0
                for batch in loader:
                    masker.step(gstep, total)
                    total_loss += step(model, batch, optimizer, masker)
                    scheduler.step()
                    gstep += 1
                losses.append(total_loss / max(len(loader), 1))
            masker_stats = dict(masker.stats)
            # Deliberately not recorded: these arms resample every step, so the
            # final mask is just the last draw, not a set the model was trained
            # to live without. Pretending otherwise would invite a meaningless
            # "aligned" comparison.
            pruned_heads = None
        finally:
            masker.remove()
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
        pruned_heads = None
    prog.done()

    if use_qat:
        model = strip_qat(model)
    model.eval()
    info = {"losses": losses, "masker_stats": masker_stats, "qat_backend": qat_backend,
            "train_seconds": round(time.time() - t0, 1), "steps": total,
            "pruned_heads": pruned_heads}
    log(f"  {key}: epoch losses {[round(x, 4) for x in losses]}"
        + (f" | masker {masker_stats}" if masker_stats else "")
        + f" | {info['train_seconds']:.0f}s")
    return model, info



# ---------------------------------------------------- test-mode presets
class PresetConflict(SystemExit):
    """Raised when a test flag would overwrite an explicit argument."""


def apply_test_preset(args, preset: dict, flag: str, argv=None) -> dict:
    """Apply a test-mode preset without overriding explicit arguments.

    ``preset`` maps argparse destinations to the values the test mode
    wants. Anything the caller passed on the command line wins -- and
    rather than let the preset win silently, or let the caller's value win
    silently, a conflict **stops the run**:

        $ exp34.py --gate --n-train 64
        --gate would set --n-train to 512, but you passed 64.

    Why refuse instead of preferring one: a test mode exists to exercise a
    specific path cheaply, and an operator who overrides part of it is
    usually trying to reach a cell the preset makes unreachable. Silently
    choosing either value produces a run that is not the one anybody
    asked for. ``--smoke`` disabling INT4, and ``--gate`` resizing a
    deliberately tiny micro-test, are the two cases this campaign already
    paid for.

    Returns the values actually applied, so a caller can record them.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    given = set()
    for tok in argv:
        if tok.startswith("--"):
            given.add(tok.split("=", 1)[0])

    conflicts = []
    for dest, value in preset.items():
        opt = "--" + dest.replace("_", "-")
        if opt in given:
            conflicts.append((opt, getattr(args, dest, None), value))
    if conflicts:
        lines = ["%s cannot be combined with an explicit value for the "
                 "arguments it sets:" % flag]
        for opt, mine, theirs in conflicts:
            lines.append("  %s would set %s to %r, but you passed %r."
                         % (flag, opt, theirs, mine))
        lines += [
            "",
            "Drop %s and pass the sizes you want, or drop the explicit "
            "argument and let %s choose. The script refuses rather than "
            "picking one, because both choices produce a run that is not "
            "the one you asked for: this is the third time a test flag has "
            "silently redefined a run in this project." % (flag, flag),
        ]
        raise PresetConflict("\n".join(lines))

    applied = {}
    for dest, value in preset.items():
        setattr(args, dest, value)
        applied[dest] = value
    return applied



#: How long a cache lock may be held before another job assumes the holder
#: died. Encoding the full 1.28M-image training cache takes about 25
#: minutes on a quiet node and longer under load, so this is deliberately
#: generous: taking a lock from a live writer is worse than waiting.
CACHE_LOCK_TIMEOUT_S = 5400
CACHE_POLL_S = 20


def _claim_cache_lock(xpath):
    """Try to become the one process that builds this cache.

    Returns the lock path if we own it, or None if someone else does. An
    exclusive create is the whole mechanism: the filesystem decides, and
    exactly one caller can win.
    """
    import os
    lock = xpath.with_suffix(".lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = None
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            pass
        if age is not None and age > CACHE_LOCK_TIMEOUT_S:
            # The holder is almost certainly dead. Say so: a silent
            # takeover would hide a crashed job.
            log("  cache lock %s is %.0f min old, taking it over"
                % (lock.name, age / 60))
            try:
                lock.unlink()
            except OSError:
                pass
            return _claim_cache_lock(xpath)
        return None
    with os.fdopen(fd, "w") as fh:
        fh.write("%s %s\n" % (os.environ.get("SLURM_JOB_ID", "local"),
                               os.getpid()))
    return lock


def _wait_for_cache(xpath, ypath, n, shape):
    """Wait for another job's encode to land, then load it.

    Returns the arrays, or None if the wait timed out -- in which case the
    caller encodes rather than hanging forever.
    """
    import numpy as np
    waited = 0
    while waited < CACHE_LOCK_TIMEOUT_S:
        if xpath.exists() and ypath.exists():
            try:
                mm = np.load(xpath, mmap_mode="r")
            except (OSError, ValueError):
                mm = None
            if mm is not None and mm.shape == shape:
                log("  another job built %s while we waited (%.0f s)"
                    % (xpath.name, waited))
                return torch.from_numpy(mm), torch.from_numpy(np.load(ypath))
        time.sleep(CACHE_POLL_S)
        waited += CACHE_POLL_S
        if waited % 300 == 0:
            log("  waiting for another job to build %s (%.0f min)"
                % (xpath.name, waited / 60))
    log("  gave up waiting for %s after %.0f min; encoding it here"
        % (xpath.name, waited / 60))
    return None


# --------------------------------------------------------------- memory
#: Filled by ``accuracy()`` on every call; read it right after.
LAST_EVAL_MEMORY: dict = {}

#: Set to a path before calling ``accuracy()`` to have it write that
#: evaluation's per-image predictions there as a compressed ``.npz``
#: (``pred`` int16, ``label`` int16). Example-level disagreement needs to
#: know WHICH images an arm loses, and collecting the argmax inside the
#: evaluation every arm already runs makes that free; a separate pass would
#: be a forward over the whole split per arm per seed. Reset to None after
#: each use so one stale path cannot silently overwrite another arm's file.
COLLECT_PREDICTIONS: "str | None" = None


def rss_bytes():
    """Current resident set size, or None where /proc is unavailable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_rss_bytes():
    """Process high-water RSS.

    Monotonic: it never falls, so inside a process that scores several
    arms it is a property of the process and not of an arm. Recorded for
    completeness, named so it cannot be misread.
    """
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except (ImportError, OSError, AttributeError):
        return None


def memory_probe(device):
    """Begin a peak-memory measurement; the returned closure ends it.

    The GPU peak is reset at the start, so what the closure reports
    belongs to the region between the two calls. Three quantities, because
    they answer different questions:

    * ``max_memory_allocated`` -- peak tensor bytes the allocator handed
      out in the region. This is the figure that scales with the arm.
    * ``max_memory_reserved`` -- the caching allocator's high-water mark
      against the driver. Always larger and history-dependent, so it is
      recorded but is not the per-arm number.
    * RSS -- current before and after, plus the process high-water mark.
    """
    on_gpu = getattr(device, "type", str(device)) == "cuda"
    if on_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = rss_bytes()

    def finish() -> dict:
        out = {"rss_bytes_before": before,
               "rss_bytes_after": rss_bytes(),
               "process_peak_rss_bytes": peak_rss_bytes(),
               "rss_note": ("process_peak_rss_bytes is a high-water mark "
                            "for the whole process and cannot be "
                            "attributed to a single arm")}
        if on_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            out["max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated())
            out["max_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved())
            out["gpu_note"] = ("allocated is the peak tensor bytes inside "
                               "the measured region, whose counter is "
                               "reset at its start; reserved is the "
                               "caching allocator's high-water mark and "
                               "depends on allocation history")
        return out

    return finish


# ------------------------------------------------------------------- scoring
@torch.no_grad()
def accuracy(model, val_split, device, use_amp: bool, batch_size: int = 128) -> float:
    """Top-1 accuracy of the model's own classifier on the val split."""
    model.eval()
    dev = next(model.parameters()).device
    correct = n = 0
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if (use_amp and dev.type == "cuda") else contextlib.nullcontext())
    # The signature stays a bare float: changing it would touch every call
    # site in two repositories. The measurement goes to a module-level
    # record that the caller reads immediately after.
    mem = memory_probe(dev)
    want_preds = COLLECT_PREDICTIONS is not None
    preds: list = []
    labels: list = []
    for x, y in batches(val_split, batch_size, workers=LOADER_WORKERS):
        with autocast:
            logits = model(pixel_values=normalize(x.to(dev))).logits
        pred = logits.argmax(-1).cpu()
        correct += int((pred == y).sum())
        n += y.shape[0]
        if want_preds:
            preds.append(pred.to(torch.int16))
            labels.append(y.to(torch.int16))
    LAST_EVAL_MEMORY.clear()
    LAST_EVAL_MEMORY.update(mem())
    LAST_EVAL_MEMORY["batch_size"] = batch_size
    LAST_EVAL_MEMORY["n_images"] = n
    if want_preds and preds:
        import numpy as _np
        out = Path(COLLECT_PREDICTIONS)
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(
            out,
            pred=torch.cat(preds).numpy().astype(_np.int16),
            label=torch.cat(labels).numpy().astype(_np.int16))
        LAST_EVAL_MEMORY["predictions"] = str(out)
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


def summary_table(results: dict, budget: dict, epochs: int, smoke: bool,
                  dataset_id: str | None = None) -> str:
    heavy, light = _box()
    cols = [c for c in CONFIGS if any(c in results.get(v[0], {}) for v in ALL_VARIANTS)]
    w0, w = 12, 10
    lines = [f"SAL+QAT Prototype - ViT-B/16 on {dataset_label(dataset_id)} "
             f"({epochs} epoch{'s' if epochs != 1 else ''}{', SMOKE' if smoke else ''})",
             heavy * (w0 + w * len(cols)),
             "",
             " " * w0 + "".join(f"{CONFIG_LABELS[c]:>{w}}" for c in cols),
             " " * w0 + "".join(f"{light * (w - 2):>{w}}" for c in cols)]

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    for key, label, _, _ in ALL_VARIANTS:
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
        for key, label, _, _ in ALL_VARIANTS[1:]:
            v = results.get(key, {}).get(KEY_CELL, {}).get("accuracy")
            if v is not None:
                deltas[key] = v - base
                lines.append(f"  {label:<10} {v * 100:6.2f}%  ({deltas[key] * 100:+.2f}pp)")
        # Membership, not a count: a run with --variants baseline drophead
        # progressive fi_drophead also yields three deltas, and the additive
        # prediction is meaningless (and a KeyError) without the SAL/QAT arms.
        if {"sal", "qat", "sal_qat"} <= deltas.keys():
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
    nf4 = [k for k, *_ in ALL_VARIANTS if "prune50_int4_nf4" in results.get(k, {})]
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
    cols = [c for c in CONFIGS if any(c in results.get(v[0], {}) for v in ALL_VARIANTS)]
    present = [v for v in ALL_VARIANTS if v[0] in results]
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
                   choices=[v[0] for v in ALL_VARIANTS], help="subset of arms to run")
    p.add_argument("--manual-qat", action="store_true",
                   help="skip torch.ao prepare_qat and use the manual STE fake-quantizer")
    p.add_argument("--no-save-models", action="store_true",
                   help="do not write checkpoints to <output>/models/ (~350MB each)")
    p.add_argument("--reuse-models", action="store_true",
                   help="load <output>/models/<variant>.pt instead of retraining when present")
    p.add_argument("--workers", type=int, default=8, help="dataset decoding workers")
    p.add_argument("--dataset", default=None,
                   help=f"HF dataset id (default: {DATASET_ID})")
    p.add_argument("--uniform-per-layer", action="store_true",
                   help="equal head quota per layer in every draw, so the trained "
                        "set is a valid slice_heads() argument and the retained "
                        "accuracy becomes a physically smaller model (exp14, "
                        "PHASE2F; the sal-torch v0.6.0 default)")
    p.add_argument("--loader-block", type=int, default=512,
                   help="read the memory-mapped split in contiguous blocks of "
                        "this many rows, shuffling block order and within-block "
                        "order. 0 for a full shuffle, which measured 3.6x slower "
                        "on a 193 GB cache. Requires the shuffled cache the "
                        "encoder writes")
    p.add_argument("--loader-workers", type=int, default=6,
                   help="DataLoader workers for the train/eval loaders when "
                        "--data-cache is used (0 elsewhere, as before)")
    p.add_argument("--data-cache", default=None,
                   help="encode the splits to uint8 .npy here and memory-map them "
                        "instead of holding them in RAM. Required for ImageNet-1k: "
                        "its train split is 193 GB as uint8")
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
    log(f"  dataset: {dataset_label(args.dataset)}")
    if args.data_cache:
        # A memory-mapped split is read from Lustre row by row; workers overlap
        # those reads with the GPU step. An in-RAM split gains nothing and pays
        # for pickling, so this stays 0 without --data-cache.
        global LOADER_WORKERS, LOADER_BLOCK
        LOADER_WORKERS = args.loader_workers
        LOADER_BLOCK = args.loader_block
    train_split, val_split, num_classes = build_data(
        sizes, args.seed, args.workers,
        dataset_id=args.dataset, cache_dir=args.data_cache)
    log(f"  {train_split[0].shape[0]} train / {val_split[0].shape[0]} val images, "
        f"{num_classes} classes")
    # The probe goes through the shared builder like every other script's,
    # so this file is not a seventh construction with its own conventions.
    n_fi = min(sizes["n_fi"], val_split[0].shape[0])
    # A uniform batch size, chosen so the probe is never ragged and never
    # empty: validate_probe refuses both, and a tiny split (the unit-test
    # fixture has twelve val images) would otherwise ask for zero batches.
    probe_bs = min(16, n_fi)
    fi_batches = build_probe(val_split[0], n_fi // probe_bs, probe_bs, device,
                             where="fi probe")
    verify_batch = {"pixel_values": fi_batches[0]["pixel_values"][:4]}

    results, training, budget = {}, {}, {}
    for key, label, mask_mode, use_qat in ALL_VARIANTS:
        if key not in args.variants:
            continue
        rule(f"Variant: {label}  (masking={mask_mode}, "
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
                key, mask_mode, use_qat, num_classes, train_split, epochs,
                args.batch_size, args.seed, device, use_amp, args.mask_ratio,
                force_manual_qat=args.manual_qat, fi_batches=fi_batches,
                uniform_per_layer=args.uniform_per_layer)
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
    payload = {
        "model": MODEL_ID,
        "dataset": args.dataset or DATASET_ID,
        "dataset_label": dataset_label(args.dataset),
        "loader_workers": LOADER_WORKERS,
        "loader_block": LOADER_BLOCK,
        "uniform_per_layer": args.uniform_per_layer,
        "smoke": args.smoke,
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
        "figures": {}, "wall_clock_s": round(time.time() - started, 1),
    }
    # The measurements go to disk BEFORE anything that merely presents them.
    # A formatting bug must never be able to destroy a finished run: twenty
    # expF jobs died inside summary_table() after 35 minutes of training each
    # and left nothing behind, because the table was written first.
    dest = outdir / "sal_qat_prototype.json"
    dest.write_text(json.dumps(payload, indent=2))
    log(f"  metrics -> {dest}")

    table = ""
    try:
        table = summary_table(results, budget, epochs, args.smoke, args.dataset)
        (outdir / "summary_table.txt").write_text(table + "\n", encoding="utf-8")
        log(f"  table   -> {outdir / 'summary_table.txt'}")
        payload["figures"] = make_figures(outdir, results, budget)
        dest.write_text(json.dumps(payload, indent=2))
    except Exception as exc:  # presentation only; the metrics are already safe
        log(f"  WARNING: table/figures failed ({type(exc).__name__}: {exc})")
        log(f"  the measurements are intact in {dest}")

    rule("Summary")
    log(table)
    log()
    log(f"wall clock: {(time.time() - started) / 60:.1f} min")
    log(f"Everything is in {outdir.resolve()} — scp that directory back.")
    return payload


if __name__ == "__main__":
    main()
