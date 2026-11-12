# exp32 — the ratio sweep, re-anchored at 2 epochs of recovery

**STUB — no results yet.** Job IDs are at the bottom; the tables arrive when
they do.

**ViT-B/16 (`google/vit-base-patch16-224`), ImageNet-1k
(`evanarlian/imagenet_1k_resized_256`), 3 seeds (42, 123, 456), full
50 000-image validation split, autocast bfloat16, batch 64.**

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

## What it should answer

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
| | | | _pending submission_ |

## Caveats, known in advance

- INT4 is `quantize_weights(model, 4)`, a simulated per-channel round-trip in
  fp32, not a 4-bit kernel. It says nothing about speed or memory.
- Head sets are deterministic given the checkpoint and **nested** (25% ⊂ 33% ⊂
  50%), so these are not three independent architectures, and every "before
  recovery" figure is one measurement repeated, not three.
- Magnitude selection only. Nothing here tests whether the knee moves under a
  different selection rule.
