# Probe — I-JEPA ViT-H/14 batch size on one MI250X GCD (gates 1b)

`experiments/probe_jepa_vram.py`, job `23499252`, `COMPLETED 0:0` in **4:31**
(0.08 GPU-hours). `.venv-tf47` (transformers 4.47.1), torch 2.5.1+rocm6.2.

`facebook/ijepa_vith14_1k`, 224×224, patch 14 → 256 patches/image. The probe
runs the **real** training step from `run_jepa_sal.make_step`: a `no_grad`
target forward, a gradient-carrying forward on patch-masked pixels
(`PATCH_MASK_RATIO = 0.40`), MSE loss, backward, `clip_grad_norm_` over all
parameters, AdamW step. Two steps per candidate, because AdamW's two fp32
moment buffers are only allocated on the first `.step()` and have to be inside
the reported peak. Each candidate gets a freshly built model and optimizer.

## Result

| batch | peak allocated | reserved | % of 68.7 GB | steady step | img/s |
|---|---|---|---|---|---|
| **64** | **32.31 GB** | 34.68 GB | **47%** | 1.40 s | **45.7** |
| 32 | 18.08 GB | 19.35 GB | 26% | 0.72 s | 44.4 |
| 16 | 10.98 GB | 11.71 GB | 16% | 0.38 s | 42.1 |

**Batch 64 fits, at 47% of VRAM. No gradient checkpointing needed.** Losses are
meaningless (random pixels) but finite and consistent at ~0.76 on all three,
which is the check that the step actually ran.

## Two things this changes

### 1. The shipped heuristic is wrong by 4x on this hardware

`run_jepa_sal._VRAM_BATCH = [(70, 32), (38, 16), (20, 8), (0, 4)]` requires
≥70 GB for batch 32. An MI250X GCD reports **68.7 GB**, which falls through to
the 38 GB bucket, so the script would have chosen **batch 16** — using 16% of
the card. The comment above that table ("the optimizer alone wants ~10GB before
a single activation is stored") is accurate: weights+grads+moments are 10.98 GB
at batch 16. The thresholds are simply pessimistic about activations.

Measured headroom per image is ~0.33 GB (from 10.98 GB at 16 to 32.31 GB at 64),
so the fixed cost is ~5.7 GB and each image costs ~0.33 GB. Batch 128 would
extrapolate to ~48 GB and probably fit, but see below for why it is not worth it.

**1b must pass `--batch-size 64` explicitly**, or be run after the table is
corrected. Leaving it to the default silently quadruples the step count.

### 2. Batch size buys almost no throughput — 45.7 vs 42.1 img/s, 8%

ViT-H/14 at 224 is already compute-saturated at batch 16 on this GCD. So the
VRAM headroom is **not** a 4x speedup; it is 8%, plus fewer optimizer steps and
a less noisy gradient. That matters for the cost estimate below, which is set by
throughput and not by batch size.

## What 1b costs, from measured throughput

2 epochs × 1 281 167 images = 2 562 334 images per run, at 45.7 img/s →
**15.6 GPU-hours per run**.

| arms × seeds | runs | training | + probe/eval | total |
|---|---|---|---|---|
| SAL + control + DropHead, 3 seeds | 9 | 140.2 h | ~5–9 h | **~145–150 GPU-h** |

That fits inside the 160 GPU-hours approved, with the DropHead arm included —
but with little margin. Dropping to one seed would cost ~50 GPU-hours and, on
exp26's evidence (sd 0.06–0.14pp across three seeds), would lose very little.

The probe excludes data loading, so this is a floor. It is a tight floor: the
memmap loader sustained **291 img/s** on ViT-B/16 in exp26 through the same code
path, which is 6x what ViT-H/14 needs, so the loader will not be the constraint.

## Caveats

- `torch.randn` pixels, not real images. VRAM does not depend on pixel values
  and the measured step time is compute, but the loss figures mean nothing and
  the throughput excludes data loading.
- **SAL's `HeadMasker` is not installed.** It is a forward pre-hook that zeroes
  slices of an activation that already exists and allocates only the mask
  vectors, so its cost is negligible — but it is not zero and it is not
  measured here.
- `peak` is `max_memory_allocated`; `reserved` is the allocator's larger
  reservation, which is what has to fit alongside fragmentation. At batch 64 the
  gap is 2.4 GB.
- Step 1 at batch 64 took **84.52 s** against 1.40 s for step 2 — that is
  MIOpen compiling the patch-embedding convolution, as documented in
  `CLAUDE.md`'s MIOpen note. Only the steady-state step feeds the estimate.
- One GCD, one model, one image size. A second GCD on the same MI250X package
  shares HBM bandwidth; nothing here measures two concurrent jobs.
