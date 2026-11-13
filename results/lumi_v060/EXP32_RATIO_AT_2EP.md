# exp32 — the ratio sweep, re-anchored at 2 epochs of recovery

**COMPLETE — 6 jobs, 3 seeds, 23.87 GPU-hours, all `COMPLETED 0:0`.** Data in
`exp32_ratios_s{42,123,456}/exp29_recovery.json` and
`exp32_ctrl_s{42,123,456}/exp31_dense_control.json`.

**ViT-B/16 (`google/vit-base-patch16-224`), ImageNet-1k
(`evanarlian/imagenet_1k_resized_256`), 3 seeds (42, 123, 456), full
50 000-image validation split, bfloat16 autocast for training and float16 autocast for scoring, batch 64.**

## Headline

**Two epochs is better than five on every arm, and pruning costs more than the
five-epoch tables said.** The dense control gains +1.11pp from the shorter
schedule against the pruned arms' +0.38 to +0.85pp, so the sweep's costs rise
to **−1.30 / −1.93 / −3.91pp** at 25 / 33 / 50% of heads. Two epochs stays the
default — cheaper and absolutely better everywhere — but the 5-epoch cost
figures must not be requoted alongside the 2-epoch recipe.

## Why

exp29 measured the recovery curve and found it **peaks at 2 epochs**, with 5
below 2 on 3/3 seeds in FP and INT4, and 10 below 1. But exp29's own ratio
sweep (Block B) ran at **5** epochs, because the cross-check against exp24
required it. So the published ratio costs — −1.03pp at 25%, −1.59pp at 33%,
−3.17pp at 50% against the matched control — are measured *past the peak*, and
exp29 says so: they are an **upper bound**, each arm would likely read 0.2-0.4pp
better at 2 epochs, the control moves too, and the net is not predictable from
there.

This re-measures the sweep **and its control** at the new default.

## No new code

exp32 is a re-run configuration, not a new experiment. The pruned arms are
`exp29_recovery_epochs.py --skip-curve --ratio-epochs 2`; the control is
`exp31_dense_control.py --epochs 2`. Running the identical harness that
produced the 5-epoch numbers is the point — the comparison is then a change of
one argument, not of a code path.

## Comparison convention

Every pruned arm is scored against **the dense control trained for the same
number of epochs**, never against the pretrained checkpoint, and the two
effects are reported separately:

- **fine-tuning gain** = dense control − pretrained — what the epochs bought.
- **pruning cost** = arm − matched dense control — what the compression cost.

The checkpoint is not a control: an arm that was fine-tuned has had epochs the
checkpoint never had, so scoring against it reports the sum of the two and can
make a model that pruning damaged look improved. Where an arm has one seed it
is scored against **that seed's own** control, not the three-seed mean (exp28
measured a 7.56pp spread from evaluation shuffling alone).

Full ladder and sources in `CONTROL_CONVENTION.md`. For ViT-B/16 on
ImageNet-1k:

| dense control | top-1 | INT4 | seeds | fine-tuning gain |
|---|---|---|---|---|
| 0 ep — pretrained | 78.44% | 77.23% | deterministic | — |
| **2 ep** | **81.554%** | **80.273%** | 3, sd 0.11 | **+3.11pp** |
| 5 ep | 80.44% | 79.01% | 3, sd 0.05 | +2.00pp |

## Results so far

### The control, 3 seeds — and the headline

| dense control, 2 ep | top-1 | INT4 | train |
|---|---|---|---|
| seed 42 | 81.676% | 80.378% | 126 min |
| seed 123 | 81.524% | 80.140% | 121 min |
| seed 456 | 81.462% | 80.300% | 127 min |
| **mean** | **81.554%** (sd 0.11) | **80.273%** | — |
| fine-tuning gain vs pretrained | **+3.11pp** | **+3.04pp** | — |

The 5-epoch control reads 80.44% (exp25, 3 seeds) for a gain of +2.00pp. **So
the dense arm, with nothing pruned, is 1.11pp better after 2 epochs than after
5.** exp29's recovery knee is not a property of recovery at all — it is this
fine-tuning recipe on this dataset, and the dense arm shows it more strongly
than any pruned arm does. That reframing is written up in
`EXP29_RECOVERY_EPOCHS.md`.

### The sweep at 2 epochs, 3 seeds

| ratio | heads | seed 42 | seed 123 | seed 456 | mean | INT4 mean |
|---|---|---|---|---|---|---|
| 25% | 36 | 80.40% | 80.21% | 80.17% | **80.258%** | 78.744% |
| 33% | 48 | 79.62% | 79.70% | 79.56% | **79.626%** | 78.003% |
| 50% | 72 | 77.81% | 77.44% | 77.68% | **77.642%** | 75.427% |

The 50% cell replicates exp29's Block A 2-epoch point (77.57%) to **0.07pp** on
an independent run — the cross-check that the two configurations are the same
measurement.

### Against matched controls, the costs grew

| ratio | 5 ep | vs 5-ep control (80.44%) | 2 ep | vs 2-ep control (81.554%) | cost change |
|---|---|---|---|---|---|
| 25% | 79.41% | −1.03pp | 80.258% | **−1.30pp** | 0.27pp worse |
| 33% | 78.85% | −1.59pp | 79.626% | **−1.93pp** | 0.34pp worse |
| 50% | 77.27% | −3.17pp | 77.642% | **−3.91pp** | 0.74pp worse |
| 25%, INT4 | 77.69% | −1.32pp | 78.744% | **−1.53pp** | 0.21pp worse |
| 33%, INT4 | 77.11% | −1.90pp | 78.003% | **−2.27pp** | 0.37pp worse |
| 50%, INT4 | 75.03% | −3.98pp | 75.427% | **−4.85pp** | 0.87pp worse |

Three seeds on every cell on both sides. The arms gained +0.85 / +0.78 / +0.38pp
from the shorter schedule; the control gained **+1.11pp**; so every cost rose,
and the INT4 column rose further than the FP one.

## What it answered

1. **Do the ratio costs shrink at 2 epochs?** No — they **grow**, by 0.27 to
   0.74pp in FP and up to 0.87pp at INT4. Both sides gain from the shorter
   schedule and the control gains more.
2. **Does the matched control improve by as much?** It improves by *more*:
   +1.11pp against the pruned arms' +0.38 to +0.85pp. exp32's framing — "if the
   control gains what the pruned arms gain, the costs are unchanged and only
   the price drops" — resolves to **the price drops and the cost rises**.
3. **Is the ordering preserved?** Yes, 25 < 33 < 50 and monotone, as at 5
   epochs.

Two epochs stays the default: cheaper, and absolutely better on every arm,
dense and pruned. But the published ratio costs must be requoted — **−1.30 /
−1.93 / −3.91pp**, not −1.03 / −1.59 / −3.17 — and every such figure has to
name the epoch count on both sides from now on.

## What it was asked to answer

1. Do the ratio costs shrink at 2 epochs, and is the ordering (25 < 33 < 50)
   preserved?
2. Does the **matched control** improve by as much? exp31 measured it at 5
   epochs (80.556% through this harness). If the control gains what the pruned
   arms gain, the *costs* are unchanged and only the price drops.
3. Is 2 epochs the right default at every ratio, or only at the 50% where
   exp29 measured the curve?

## Jobs

| seed | arm | job | state |
|---|---|---|---|
| 42 | control, dense + 2 ep | `23670468` | **COMPLETED** 2:13:53 |
| 123 | control, dense + 2 ep | `23670469` | **COMPLETED** 2:09:39 |
| 456 | control, dense + 2 ep | `23670470` | **COMPLETED** 2:13:22 |
| 42 | ratios 0.25 / 0.33 / 0.50 @ 2 ep | `23670464` | **COMPLETED** 5:36:39 |
| 123 | ratios | `23670465` | **COMPLETED** 5:36:39 |
| 456 | ratios | `23670466` | **COMPLETED** 5:44:21 |
| — | smoke | `23670101` | COMPLETED 0:05:07 |

## Caveats, known in advance

- Precision, stated exactly: **training and recovery run under `bfloat16` autocast; scoring runs under `float16` autocast** (`run_sal_qat_prototype.py:accuracy`). The two differ, and earlier revisions of this page said "autocast bfloat16" for both. fp16 carries more mantissa than bf16 and its range is ample for a ViT on normalised inputs, so this is a labelling fix and not a re-measurement -- but it also means the INT4 column is **INT4-rounded weights scored under fp16 autocast**, not under fp32.
- INT4 is `quantize_weights(model, 4)`, a simulated per-channel round-trip in
  fp32, not a 4-bit kernel. It says nothing about speed or memory.
- Head sets are deterministic given the checkpoint and **nested** (25% ⊂ 33% ⊂
  50%), so these are not three independent architectures, and every "before
  recovery" figure is one measurement repeated, not three.
- Magnitude selection only. Nothing here tests whether the knee moves under a
  different selection rule.
- **The 2-epoch knee is a property of this recipe**, measured now on the dense
  arm as well as the pruned ones. Nothing here says it transfers to another
  dataset, another LR schedule, or a model not already fine-tuned for its task.
- The three controls vary only data order and the evaluation shuffle; the
  pretrained row is one deterministic measurement, not three.
