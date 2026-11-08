# exp28 — I-JEPA ViT-H/14 on ImageNet-1k: a recipe failure, not a SAL result

`experiments/exp28_jepa_1b.py`, `.venv-tf47` (transformers 4.47.1), torch
2.5.1+rocm6.2, one MI250X GCD per job.

**Read this before quoting any number from `results/jepa1b*`.** Every arm in this
experiment — including the unperturbed control — lost most of its representation
quality during two epochs of continued pretraining. The control being broken is
what makes the experiment uninterpretable as a test of SAL or DropHead. Nothing
here says either method is good or bad; it says the training recipe is wrong for
this model.

## Protocol

`facebook/ijepa_vith14_1k`, **630.8M parameters**, 32 layers × 16 heads = 512
heads, patch 14 → 256 patches per 224×224 image.

**ImageNet-1k (resized 256px, `evanarlian/imagenet_1k_resized_256`)**,
1 281 167 train / 50 000 val, 1000 classes. Two epochs of continued I-JEPA
pretraining — a `no_grad` target forward, a gradient-carrying forward on
patch-masked pixels (`PATCH_MASK_RATIO = 0.40`), MSE loss, `clip_grad_norm_`,
AdamW — from `scripts/run_jepa_sal.py:make_step`, i.e. the same step the v0.5.1
I-JEPA results used. Batch 64, chosen from `PROBE_1B_VRAM.md` (32.3 GB peak,
47% of the GCD).

Three arms, identical data, optimiser, schedule and epoch count, differing only
in masking:

| arm | masking | `prune_events` |
|---|---|---|
| `baseline` (printed as `control`) | none | — |
| `sal` | accumulate-and-hold, 256 of 512 heads, uniform per layer | 256 |
| `drophead` | mask resampled every step, same 50% quota | 0 |

Three training seeds (42, 123, 456). Accuracy is `sal.evaluation.linear_probe`
on **frozen features**: I-JEPA is self-supervised and has no 1000-class head, so
this is **not** ImageNet top-1 and is not comparable with exp23–27. The probe is
fitted on 5000 images for 1000 classes — 5 images per class.

Jobs, all `COMPLETED 0:0` unless stated:

| seed | baseline | sal | drophead | eval |
|---|---|---|---|---|
| 42 | `23515576` (15:45:32) | `23516504` (16:09:35) | `23516505` (20:01:36) | `23523239` (11:36:09) |
| 123 | `23523367` (15:52:31) | `23523368` (16:03:41) | `23523369` (19:54:16) | `23523370` **RUNNING** |
| 456 | `23523371` (15:49:53) | `23523372` (16:37:07) | `23523373` (20:07:20) | `23523374` **RUNNING** |

Gates and probes: `23500572` (1:22:53), `23499252` VRAM probe (4:31),
`23511821` noise probe (24:23), smoke `23511932`-`23511934` + `23515525`
(1:11:05), plus `23511935` eval **TIMEOUT** at 1:00:27 and `23515577`/`23515578`
**CANCELLED** at 36:05 each.

**Cost: ~157.5 GPU-hours of training and ~22 GPU-hours of evaluation so far**,
~181 GPU-hours in all including gates and the two evals still running. The brief
for this document estimated ~70 + ~47; the measured figure is 2.3x that, because
each arm is a 15–20 hour job and there are nine of them.

## Result — seed 42 (complete)

Linear probe on frozen features, 2500 validation images, 1000 classes. Chance is
0.1%.

| arm | probe | INT4 | vs `original` |
|---|---|---|---|
| **`original`** — pretrained, no training of any kind | **48.60%** | 39.40% | — |
| **`baseline`** — 2 epochs, no masking | **16.64%** | 15.64% | **−31.96pp** |
| **`drophead`** — 2 epochs, mask resampled per step | **3.04%** | 3.08% | **−45.56pp** |
| **`sal`** — 2 epochs, accumulate-and-hold | **2.32%** | 2.52% | **−46.28pp** |

Every pruning cell sits inside the noise of its own collapsed dense row:

| head set | sal | baseline (`ctrl`) | drophead |
|---|---|---|---|
| dense | 2.32 | 16.64 | 3.04 |
| random 33% (20 draws) | 2.56 ± 0.28 | 17.93 ± 3.59 | 3.29 ± 0.23 |
| random 50% (20 draws) | 2.32 ± 0.66 | 12.15 ± 2.72 | 3.49 ± 0.36 |
| magnitude 33% | 1.60 | 17.08 | 3.16 |
| magnitude 50% | 1.64 | 12.08 | 3.36 |
| SAL's aligned 50% set | 2.72 | 17.68 | 3.68 |

Note what the aligned row does **not** show. In exp25, on ViT-B/16, SAL read
76.51% on its own trained-for set against 20.33% for the baseline on the
identical set. Here SAL reads **2.72%** on its own set against the baseline's
**17.68%**. Removing SAL's own 256 heads from the *unmasked* checkpoint is 15pp
better than the checkpoint that trained for exactly that set. There is no reading
of that as a SAL result; it is what a destroyed representation looks like.

Seeds 123 and 456 are being scored now. Their live logs already report
`sal+random-33%: probe 2.58% ± 0.36`, `sal+random-50%: probe 2.86% ± 0.58`
(seed 123) and `sal+random-50%: probe 1.96% ± 0.39` (seed 456), so the collapse
is reproducing. **This document must be updated with both seeds when the two
jobs finish** — until then the table above is n=1.

## Diagnosis

### 1. The loss goes down while the representation is destroyed

Per-epoch training loss, and the resulting probe:

| arm | epoch 1 | epoch 2 | final probe |
|---|---|---|---|
| sal | 0.14274 | **0.01369** | 2.32% |
| baseline | 0.14964 | 0.03132 | 16.64% |
| drophead | 0.53407 | 0.24393 | 3.04% |

Every arm's loss falls. Every arm's features end up far below the checkpoint it
started from. The I-JEPA objective is a predictive MSE between an online encoder
and a target encoder, and the degenerate solution — both collapsing toward a
low-variance output — scores extremely well on it. A loss of 0.0137 is not a
better model, it is a more complete collapse.

Two things this does *not* establish, which matter for honesty:

- **The loss does not order the arms.** DropHead has the *highest* final loss
  (0.244) and still ends at 3.04%. So "lower loss ⇒ worse features" holds for
  `sal` against `baseline` and fails for `drophead`, which is perturbed in a way
  that keeps it from settling anywhere. The clean statement is that **loss
  progress on this objective is decoupled from representation quality here**, not
  that it is inversely proportional to it.
- **There is no intermediate measurement.** The probe was run once, at the end.
  Nothing here shows *when* the collapse happened or whether it was monotone. A
  per-epoch (or per-1000-step) probe is the missing instrument and is cheap
  relative to a 16-hour training job.

### 2. SAL is the most collapsed because it is the most perturbed

The three arms order exactly by how much structural damage they carry while the
encoder is trying to track a moving target: 256 heads permanently off (sal) <
256 heads off but resampled (drophead) < nothing off (baseline). Under a recipe
that is already collapsing, more perturbation collapses further. That is the
expected interaction and it says nothing about either method's value on a recipe
that works — which is what exp27 and exp25 measure, on ViT-B/16, with a control
that holds.

### 3. The probe is not at fault

This was the first thing checked, because a 48 → 2% reading on a 5-images-per-
class probe invites exactly that suspicion.

| measurement | source | probe on the pretrained checkpoint |
|---|---|---|
| exp28's own `original` row | `results/jepa1b/exp28_jepa_1b.json` | **48.60%** |
| `probe_jepa_noise`, 5 disjoint 5000-image blocks | `PROBE_1B_VRAM.md`'s companion, job `23511821` | **49.09% ± 0.58** (48.40–49.84), kNN 44.93% |

The two agree to 0.5pp, and the probe's own spread over which 5000 images it is
fitted on is **0.58pp** — verdict `PROBE IS STABLE ENOUGH` in that JSON. A
measurement instrument with 0.6pp of noise did not manufacture a 32pp drop in
the control arm. The collapse is in the checkpoints.

### 4. The slicing path is not at fault either

Every `-sliced` row agrees with its masked equivalent to
`max_abs_diff_vs_masked ≈ 7e-06` (relative 1.3e-06), on all twelve sliced cells,
`slicing_errors: {}`. `slice_heads()` works correctly on a 631M I-JEPA encoder;
it is removing heads from an already-destroyed model. Measured incidentally:
removing 33% of heads uniformly gives **1.00x on GPU, 1.10x on CPU** at 89.6% of
the parameters, consistent with v0.5.1.

## Conclusion

**This run measures nothing about SAL or DropHead.** Its control arm lost 31.96pp
of probe accuracy with no masking at all, so there is no reference against which
the masked arms mean anything. The three findings that survive are:

1. **Two epochs of continued I-JEPA pretraining at this configuration destroy the
   pretrained representation.** The recipe — learning rate, schedule, and in
   particular the target-encoder EMA — has to be adapted for a self-supervised
   objective before any arm comparison is run on it. `make_step` was validated at
   v0.5.1 on short I-JEPA runs; it does not survive two full ImageNet-1k epochs.
2. **The measurement chain is sound.** Probe (±0.58pp), slicing (1e-06 against
   masking), and the dataset path all check out. The instruments are not the
   problem.
3. **Gate on the control arm, not on completion.** The smoke jobs ran in 90
   seconds and passed; the gate ran in 1:22:53 and passed; both exercise the code
   path and neither can see a representational collapse that takes 16 hours of
   real training to produce. **A cheap probe of the control checkpoint after ~1000
   steps would have caught this before 157 GPU-hours were spent** — which is the
   same lesson as `CLAUDE.md`'s "fail loudly on a non-finite loss", one level up:
   fail loudly when the *control* arm regresses against its own starting point.

What this does not do is weaken the ViT-B/16 conclusions. exp24, exp25, exp26 and
exp27 all have controls that hold, three seeds, the full 50 000-image validation
split and a supervised top-1 metric. They stand on their own; exp28 neither
supports nor contradicts them.

# Caveats

- **n=1 as written.** Only seed 42 is scored. Seeds 123 and 456 are in flight
  (`23523370`, `23523374`) and their partial logs agree; the tables must be
  refreshed when they land.
- **The accuracy column is a linear probe on frozen features, fitted on 5000
  images for 1000 classes** — 5 per class. It is not ImageNet top-1 and is not
  comparable with exp23–27, which are supervised top-1 on 50 000 images.
  Validation here is 2500 images.
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
  corrected recipe reproduces v0.5.1's I-JEPA gains at this scale is **not
  measured** — that is the experiment this one failed to be.
