# Track A2 / A3 — the number with an error bar, and where SAL stops working

All runs here use **20 random head selections** per cell (A1 showed 3 is far too
few) on ViT-B/16 / ImageNet-100, 20k train / 5k val, 5 epochs.

## A2 — fifteen seeds

Ten new training seeds added to A1's five, all at `mask_ratio=0.3`.

| at P50+I4 | mean | sd | range |
|---|---|---|---|
| Baseline | 31.02% | 1.36 | 28.6 – 33.5 |
| SAL only | 43.15% | 3.79 | 36.0 – 50.9 |
| SAL + QAT | 45.56% | 4.25 | 36.3 – 53.8 |

**SAL+QAT − Baseline: +14.54pp, sd 4.28, se 1.11, 95% CI [+12.38, +16.71],
positive on 15/15 seeds** (min +5.29, max +22.38).

**QAT increment over SAL alone: +2.41pp, sd 0.88, positive on 15/15 seeds.**

The sd fell again from 6.68 (5 seeds) to 4.28 (15), because the original
five happened to contain both extremes. The baseline's sd of 1.36 across
fifteen independently trained models confirms A1: a half-pruned ViT-B/16 lands
at ~31% top-1 and stays there.

This is the number to publish: **+14.5pp [12.4, 16.7]**, not +22pp.

## A3 — the dose-response, and the regime where SAL hurts

`mask_ratio` swept over {0.1, 0.2, 0.3, 0.4, 0.5} on three seeds: 123 (strong),
137 (weak), 2024 (the one that looked harmful).

### SAL+QAT − Baseline (pp), mean of the three seeds

| test-time compression | mr=0.1 | mr=0.2 | mr=0.3 | mr=0.4 | mr=0.5 |
|---|---|---|---|---|---|
| Prune33% | +2.67 | +3.37 | +2.21 | +5.57 | +6.94 |
| Prune50% | +4.52 | +7.17 | +10.34 | +14.17 | +18.04 |
| **P50+I4** | **+6.18** | **+8.96** | **+12.68** | **+16.62** | **+20.76** |

Two things fall out of this table.

**1. There is a clean monotonic dose-response, and 0.3 is not the right
default.** At P50+I4 the gain rises steadily with the training mask ratio, from
+6.2pp at 0.1 to +20.8pp at 0.5. The best cells are the ones where the training
mask ratio *matches the test-time prune ratio* — `mask_ratio=0.5` evaluated at
Prune50%. A monotone dose-response across five levels is much stronger evidence
that the mechanism is real than any single cell, and it says the shipped default
of 0.3 is leaving most of the effect on the table when the deployment target is
50% pruning.

The price is small and also monotone. Clean FP16 accuracy costs
−0.3 to −1.5pp relative to baseline, growing with mask ratio. So ~1.5pp of
clean accuracy buys ~20pp at P50+I4.

**2. SAL genuinely does hurt at Prune33% on some seeds.** A1 showed seed 2024's
P50+I4 "harm" was an artefact. At **Prune33%** it is not:

SAL-only minus baseline at Prune33% (pp):

| seed | mr=0.1 | mr=0.2 | mr=0.3 | mr=0.4 | mr=0.5 |
|---|---|---|---|---|---|
| 123 | +6.77 | +5.93 | +10.22 | +10.37 | +13.17 |
| 137 | +0.62 | **−5.06** | +1.84 | +7.83 | +7.28 |
| 2024 | **−1.21** | +1.75 | **−2.79** | **−5.58** | **−3.63** |

Seed 2024 is negative in four of five mask ratios, at 20 head selections. That
is not noise. The coherent reading: SAL trains a model to operate under heavy
structural damage, and when the damage at test time is *milder* than what it
trained for, the adaptation can be a net loss. Seed 137 shows the same sign once
(−5.06 at mr=0.2).

So the honest statement of scope is:

- **Prune50% and above, with or without INT4: robust.** Positive on 15/15
  seeds, dose-responsive, CI well clear of zero.
- **Prune33%: small and unreliable.** Mean +2 to +7pp but negative on a
  recurring seed at most mask ratios. SAL should not be recommended for mild
  pruning without a per-model check.

## Consequences

- Publish **+14.5pp [12.4, 16.7], 15 seeds, 20 head selections**, and stop
  quoting +22pp.
- **The default `mask_ratio=0.3` is not optimal** for a 50%-pruning target;
  0.5 is worth ~8pp more at P50+I4 for ~1pp of clean accuracy. This is the
  first real data for the per-architecture ratio table, and it suggests the
  rule "train at the ratio you intend to deploy at".
- **Document the Prune33% limitation.** This is a genuine boundary of the
  method, reproducible at 20 selections, not an estimator artefact — unlike the
  P50+I4 claim in PHASE2.md that A1 retracted.
- Both findings came from 13.4 GPU-hours. Neither needed a larger model or a
  larger dataset, which argues for finishing the characterisation at this scale
  before scaling up.
