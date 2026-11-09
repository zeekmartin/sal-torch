# exp28 — I-JEPA ViT-H/14 on ImageNet-1k: a recipe failure, not a SAL result

`experiments/exp28_jepa_1b.py`, `.venv-tf47` (transformers 4.47.1), torch
2.5.1+rocm6.2, one MI250X GCD per job.

**Read this before quoting any number from `results/jepa1b*`.** Every arm in this
experiment — including the unperturbed control — lost most of its representation
quality during two epochs of continued pretraining, on all three seeds. The
control being broken is what makes the experiment uninterpretable as a test of
SAL or DropHead. Nothing here says either method is good or bad; it says the
training recipe is wrong for this model.

**The follow-up identifies the cause.** exp28b re-ran the same model and the same
data with a **frozen target encoder**, LR 5e-6, 10% warmup and cosine decay. Its
control arm reads **47.52% after a full epoch of 1 281 167 images, against the
48.60% pretrained reference — a 1.08pp drop** where exp28's control lost ~32pp.
The collapse documented here follows from the training objective taking its
target from the live weights, not from the data, the scale or the masking.

## Protocol

`facebook/ijepa_vith14_1k`, **630.8M parameters**, 32 layers × 16 heads = 512
heads, patch 14 → 256 patches per 224×224 image.

**ImageNet-1k (resized 256px, `evanarlian/imagenet_1k_resized_256`)**,
1 281 167 train / 50 000 val, 1000 classes. Two epochs of continued I-JEPA
pretraining — a `no_grad` target forward, a gradient-carrying forward on
patch-masked pixels (`PATCH_MASK_RATIO = 0.40`), MSE loss, `clip_grad_norm_`,
AdamW — from `scripts/run_jepa_sal.py:make_step`, i.e. the same step the v0.5.1
I-JEPA results used. **The target forward uses the same live weights that are
being trained**; there is no separate target encoder and no EMA. LR 1e-5, no
warmup, no scheduler. Batch 64, chosen from `PROBE_1B_VRAM.md` (32.3 GB peak,
47% of the GCD).

Three arms, identical data, optimiser, schedule and epoch count, differing only
in masking:

| arm | masking | `prune_events` |
|---|---|---|
| `baseline` (printed as `control`) | none | — |
| `sal` | accumulate-and-hold, 256 of 512 heads, uniform per layer | 256 |
| `drophead` | mask resampled every step, same 50% quota | 0 |

Three training seeds (42, 123, 456), **all three now scored**. Accuracy is
`sal.evaluation.linear_probe` on **frozen features**: I-JEPA is self-supervised
and has no 1000-class head, so this is **not** ImageNet top-1 and is not
comparable with exp23–27. The probe is fitted on 5000 images for 1000 classes —
5 images per class — and validated on 2500.

Jobs, all `COMPLETED 0:0` unless stated:

| seed | baseline | sal | drophead | eval |
|---|---|---|---|---|
| 42 | `23515576` (15:45:32) | `23516504` (16:09:35) | `23516505` (20:01:36) | `23523239` (11:36:09) |
| 123 | `23523367` (15:52:31) | `23523368` (16:03:41) | `23523369` (19:54:16) | `23523370` (11:43:25) |
| 456 | `23523371` (15:49:53) | `23523372` (16:37:07) | `23523373` (20:07:20) | `23523374` (11:37:41) |

Gates and probes: `23500572` (1:22:53), `23499252` VRAM probe (4:31),
`23511821` noise probe (24:23), smoke `23511932`-`23511934` + `23515525`
(1:11:05), plus `23511935` eval **TIMEOUT** at 1:00:27 and `23515577`/`23515578`
**CANCELLED** at 36:05 each.

**Cost: 155.5 GPU-hours of training** (the sum of the nine arms' recorded
`train_seconds`), **35.0 GPU-hours of evaluation** and **~5.3 GPU-hours** of
gates and probes — **~196 GPU-hours in all**. The brief for this document
estimated ~70 + ~47; the measured figure is 2.4x that, because each arm is a
15–20 hour job and there are nine of them.

## Result — three seeds

Linear probe on frozen features, 2500 validation images, 1000 classes. Chance is
0.1%. Each seed's `original` row is measured **inside that seed's own eval job,
on that seed's own probe split**, so the last column is a within-seed comparison.

| arm | s42 | s123 | s456 | **mean** | sd | vs own `original` |
|---|---|---|---|---|---|---|
| **`original`** — pretrained, no training | 48.60 | 48.56 | 41.04 | **46.07** | 4.35 | — |
| **`baseline`** — 2 epochs, no masking | 16.64 | 15.68 | 13.64 | **15.32** | 1.53 | −31.96 / −32.88 / −27.40 |
| **`drophead`** — 2 epochs, resampled | 3.04 | 3.44 | 3.12 | **3.20** | 0.21 | −45.56 / −45.12 / −37.92 |
| **`sal`** — 2 epochs, accumulate-and-hold | 2.32 | 3.44 | 2.32 | **2.69** | 0.65 | −46.28 / −45.12 / −38.72 |

INT4 on the same cells: `original` 42.41 ± 4.13, `baseline` 14.80, `drophead`
3.12, `sal` 2.56. Quantisation costs nothing measurable once the representation
is gone.

Every pruning cell sits inside the noise of its own collapsed dense row. Means
over the three seeds, with the across-seed sd:

| head set | sal | baseline (`ctrl`) | drophead |
|---|---|---|---|
| dense | 2.69 ± 0.65 | 15.32 ± 1.53 | 3.20 ± 0.21 |
| random 33% (20 draws/seed) | 2.47 ± 0.17 | 16.66 ± 1.21 | 3.31 ± 0.26 |
| random 50% (20 draws/seed) | 2.38 ± 0.45 | 10.92 ± 1.08 | 3.35 ± 0.26 |
| magnitude 33% | 1.95 ± 0.30 | 16.15 ± 1.45 | 3.08 ± 0.25 |
| magnitude 50% | 1.93 ± 0.38 | 11.08 ± 0.87 | 3.28 ± 0.29 |
| SAL's aligned 50% set | 2.12 ± 0.55 | 14.39 ± 3.59 | 3.33 ± 0.38 |

Note what the aligned row does **not** show. In exp25, on ViT-B/16, SAL read
76.51% on its own trained-for set against 20.33% for the baseline on the
identical set. Here SAL reads **2.12%** on its own set against the baseline's
**14.39%**, on 3/3 seeds (2.72 vs 17.68, 1.64 vs 14.92, 2.00 vs 10.56). Removing
SAL's own 256 heads from the *unmasked* checkpoint is 12.3pp better than the
checkpoint that trained for exactly that set. There is no reading of that as a
SAL result; it is what a destroyed representation looks like.

## Diagnosis

### 1. The loss goes down while the representation is destroyed

Per-epoch training loss, three-seed means, and the resulting probe:

| arm | epoch 1 | epoch 2 | final probe |
|---|---|---|---|
| sal | 0.1513 ± 0.0096 | **0.0478 ± 0.0318** | 2.69% |
| baseline | 0.1495 ± 0.0001 | **0.0312 ± 0.0001** | 15.32% |
| drophead | 0.5340 ± 0.0004 | 0.2438 ± 0.0001 | 3.20% |

Every arm's loss falls by a factor of 2–10. Every arm's features end up far below
the checkpoint it started from. The objective is an MSE between a
gradient-carrying forward and a `no_grad` forward **of the same weights**, so it
has no fixed point: if the encoder drifts toward a low-variance output, both
sides of the comparison drift together and the loss falls without any
representation being learned. A loss of 0.031 is not a better model.

Two things the three-seed data corrects, which matter for honesty:

- **The loss does not order the arms — and at n=3 it does not even order them
  the wrong way.** On seed 42 alone SAL had the lowest final loss (0.0137) and
  the worst probe, which invited a "lower loss ⇒ more collapse" reading. Across
  three seeds SAL's epoch-2 loss is 0.0137 / 0.0767 / 0.0531 — mean 0.0478,
  **above** the baseline's 0.0312 — while the baseline has both the **lowest**
  loss and the **best** features. That reading is withdrawn. The only statement
  the data supports is that **loss progress on this objective is decoupled from
  representation quality**, in either direction.
- **There is no intermediate measurement.** The probe was run once, at the end.
  Nothing here shows *when* the collapse happened or whether it was monotone. A
  per-epoch probe is the missing instrument; exp28b adds it.

Note also the variance structure: the baseline and drophead losses are identical
across seeds to four decimal places (sd 0.0001), while SAL's epoch-2 loss spans
0.0137–0.0767. The accumulate-and-hold schedule is the only stochastic
difference between the arms, and SAL is the only arm whose optimisation is
seed-dependent at all.

### 2. SAL is the most collapsed because it is the most perturbed

The arms order by how much structural damage they carry while the encoder tracks
a target computed from itself: 256 heads permanently off (sal) < 256 heads off
but resampled (drophead) < nothing off (baseline). Under a recipe that is already
collapsing, more perturbation collapses further. That is the expected interaction
and it says nothing about either method's value on a recipe that works — which is
what exp27 and exp25 measure, on ViT-B/16, with a control that holds.

### 3. The probe did not manufacture the collapse, but it is not as stable as the n=1 version of this document claimed

This was the first thing checked, because a 48 → 2% reading on a
5-images-per-class probe invites exactly that suspicion.

| measurement | source | probe on the pretrained checkpoint |
|---|---|---|
| exp28's own `original` row, seed 42 | `results/jepa1b/exp28_jepa_1b.json` | **48.60%** |
| seed 123 | `results/jepa1b_s123/exp28_jepa_1b.json` | **48.56%** |
| seed 456 | `results/jepa1b_s456/exp28_jepa_1b.json` | **41.04%** |
| `probe_jepa_noise`, 5 disjoint 5000-image blocks | job `23511821` | **49.09% ± 0.58** (48.40–49.84), kNN 44.93% |

**Correction.** The n=1 version cited that ±0.58pp as the probe's stability. It is
not: it is the spread over which *disjoint block of one fixed shuffle* the probe
is fitted on. The three eval jobs use three different data shuffles, and across
them the same untrained checkpoint reads **41.04–48.60% — a 7.56pp spread, 13x
the noise-probe estimate**. The probe is therefore **not** a ±0.6pp instrument in
absolute terms, and no cross-seed difference smaller than ~8pp should be read off
it. The INT4 column of that row behaves the same way (39.40 / 47.12 / 40.72).

What this does not touch is the conclusion. Each seed's control is compared with
its *own* `original`, measured in the same job on the same split, and those drops
are **−31.96, −32.88 and −27.40pp**. A 7.6pp instrument did not manufacture a
27pp within-split drop reproduced on 3/3 seeds. The collapse is in the
checkpoints.

### 4. The slicing path is not at fault either

Every `-sliced` row agrees with its masked equivalent to
`max_abs_diff_vs_masked ≈ 7e-06` (relative 1.3e-06), on all twelve sliced cells
of all three seeds, `slicing_errors: {}` in every run. `slice_heads()` works
correctly on a 631M I-JEPA encoder; it is removing heads from an already-destroyed
model. Measured incidentally: removing 33% of heads uniformly gives **1.00x on
GPU, 1.10x on CPU** at 89.6% of the parameters, consistent with v0.5.1.

## Conclusion

**This run measures nothing about SAL or DropHead.** Its control arm lost 27–33pp
of probe accuracy with no masking at all, on 3/3 seeds, so there is no reference
against which the masked arms mean anything. The findings that survive are:

1. **The v0.5.1 I-JEPA step does not survive two full ImageNet-1k epochs**, and
   the reason is structural rather than hyperparametric: `make_step` computes its
   target from the weights it is training. exp28b changed exactly that — a frozen
   target encoder, plus LR 5e-6 with warmup and cosine decay — and its control arm
   holds to within **1.08pp** over a full epoch of the same data. That is the
   controlled comparison this document could not make.
2. **The measurement chain is sound, with one caveat now quantified.** Slicing
   (1e-06 against masking) and the dataset path check out. The probe is reliable
   *within* a data shuffle and moves by up to 7.6pp *across* shuffles; every
   conclusion here rests on a within-shuffle comparison.
3. **Gate on the control arm, not on completion.** The smoke jobs ran in 90
   seconds and passed; the gate ran in 1:22:53 and passed; neither can see a
   representational collapse that takes 16 hours of real training to produce.
   **A cheap probe of the control checkpoint after one epoch would have caught
   this before 196 GPU-hours were spent** — the same lesson as `CLAUDE.md`'s
   "fail loudly on a non-finite loss", one level up: fail loudly when the
   *control* arm regresses against its own starting point. exp28b implements it
   as a sentinel file the sibling arms poll.

What this does not do is weaken the ViT-B/16 conclusions. exp24, exp25, exp26 and
exp27 all have controls that hold, three seeds, the full 50 000-image validation
split and a supervised top-1 metric. They stand on their own; exp28 neither
supports nor contradicts them.

# Caveats

- **The accuracy column is a linear probe on frozen features, fitted on 5000
  images for 1000 classes** — 5 per class. It is not ImageNet top-1 and is not
  comparable with exp23–27, which are supervised top-1 on 50 000 images.
  Validation here is 2500 images.
- **The probe's absolute level moves with the data shuffle** (41.04–48.60% on the
  same untrained checkpoint across the three eval jobs). Compare within a seed,
  never across.
- **INT4 is the simulated per-channel quantize/dequantize round-trip in fp32**,
  not a 4-bit kernel. bitsandbytes NF4 is CUDA-only and absent on this ROCm
  image, so native INT4 is unavailable on this hardware. The column measures the
  accuracy cost of 4-bit weight rounding and nothing about speed or memory.
- **Every row but the `-sliced` ones masks heads rather than removing them**, so
  parameter count and arithmetic are unchanged and the latency columns correctly
  show no gain.
- `drophead` is scored on head sets it never saw, because per-step resampling
  gives it no "own" set. That asymmetry is the point elsewhere and is irrelevant
  here, since every arm is collapsed.
- The two `CANCELLED` jobs (`23515577`, `23515578`) were killed at 36 minutes by
  the scheduler and relaunched as `23516504`/`23516505`; nothing from them is in
  any table.
- One model, one dataset, one recipe, one mask ratio (0.5), two epochs. Whether a
  corrected recipe reproduces v0.5.1's I-JEPA *gains* at this scale is still
  **not measured** — exp28b establishes that the control survives, which is a
  precondition for that question, not an answer to it.
