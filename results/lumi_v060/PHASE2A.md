# Track A1 — where the variance actually was

exp04 left a headline of +16.13pp with sd 11.07 at P50+I4, a standard deviation
two thirds of the mean. Before buying more results, A1 asks how much of that
spread is the *measurement* rather than the method.

Each compression cell in exp04 averaged only **3** random head selections. A1
re-scores **the same five checkpoints** at **20** selections. No retraining:
`--reuse-models` loads the saved weights, so this is pure evaluation and cost
about 0.5 GPU-hours for all five seeds.

## Result: most of it was the measurement, and the baseline is rock-solid

| at P50+I4 | 3 selections | 20 selections |
|---|---|---|
| **baseline** sd | 5.58 | **0.71** |
| **baseline** spread (max−min) | 12.8pp | **1.6pp** |
| SAL-only sd | 12.00 | 5.84 |
| SAL+QAT sd | 13.46 | 6.84 |
| **delta** mean | +16.13pp | **+14.45pp** |
| **delta** sd | 11.07 | **6.68** |
| **delta** minimum over seeds | +0.57pp | **+5.29pp** |
| QAT increment mean | +2.65pp | +2.52pp |
| QAT increment sd | 2.67 | **1.32** |
| QAT increment positive on | 4/5 seeds | **5/5 seeds** |

The headline number barely moved (+16.1 → +14.5pp) but **the error bar shrank
by 40%**, and three things changed qualitatively.

**The baseline was never unstable.** Its P50+I4 accuracy across the five seeds
is 32.5 / 31.4 / 32.6 / 32.2 / 31.0% — a 1.6pp spread, sd 0.71. In exp04 the
same five checkpoints appeared to span 23.6–36.5%. That entire 12.8pp range was
noise from averaging 3 head selections. A ViT-B/16 that has lost half its heads
lands at ~32% top-1, reliably, and PHASE2.md's reading that "the baseline and
SAL arms are each unstable" was wrong about the baseline.

**Seed 2024 is not a failure case.** Its delta went from +0.57pp to **+5.29pp**,
and the minimum delta over all five seeds rose from +0.57 to +5.29pp. SAL's
benefit is therefore *more* reliably positive than the 3-selection estimate
suggested, even though the mean came down. The "first observation of SAL doing
net harm", flagged in PHASE2.md, does not survive a better estimator — a
caution against reading single cells off a noisy table, which is exactly what
that note did. Whether 2024 is merely a weak seed rather than a harmful one is
what A3's `mask_ratio` sweep now tests.

**The QAT increment is real and small.** +2.52pp mean, sd 1.32, positive on
5/5 seeds (it was 4/5 with one seed at −1.68pp). This is now the most solid
quantity in the experiment.

## What remains is genuinely SAL's variance

SAL-only P50+I4 still spans 36.0–50.9% (sd 5.84) with the baseline pinned at
~32%. Halving the sd did not remove it. So *how much* a model gains from
progressive head masking really does depend on the training seed, by roughly
±6pp — that part is the method, not the instrument, and it is the honest error
bar to publish.

## Consequences

- **Report `+14.5pp ± 6.7` (5 seeds, 20 head selections)**, not +22pp and not
  +16pp. The spread is a property of SAL, the earlier larger spread was not.
- **`--prune-seeds 3` is too few for any published cell.** It is the cheapest
  variance in the experiment — evaluation only, no training — and 20 cuts the
  error bar by 40%. Nothing should be quoted off 3 again.
- **The v0.5.0 NVIDIA figures are themselves 3-selection estimates.** They
  cannot be differenced against a 20-selection run without mixing a platform
  effect into a sampling effect, so `aggregate_exp04.py` now refuses that
  comparison unless both sides used 3. The platform check stands on the
  like-for-like 3-vs-3 run in PHASE2.md: mean difference **+0.99pp**.
- PHASE2.md's two strongest readings — unstable baseline, harmful seed — were
  both artefacts of the estimator. They are left in place rather than quietly
  edited, with this file as the correction.
