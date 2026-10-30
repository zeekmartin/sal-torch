# Track C — the controls beat SAL, decisively

The question: is SAL's compression resilience actually SAL's? Two controls, both
using the *same* masking mechanism (a real `sal.HeadMasker`, same pre-hook, same
per-head slicing) with only `step()` changed:

- **`drophead`** — classic stochastic head dropout. `mask_ratio` of heads masked
  every step, **resampled every step**, on from step 0, never held.
- **`nonaccum`** — SAL's exact schedule with accumulation removed. The pruned
  *count* follows the identical ramp over the identical window; only the
  identities are resampled each step, so no head stays dead.

ViT-B/16 / ImageNet-100, 10 seeds, 20 random head selections per cell, two mask
ratios. The 10 seeds are ones already measured, and the `baseline`/`sal` arms
reproduced the published 15-seed run to within 0.69pp mean (delta +12.40 →
+12.42pp), so the apparatus is not in question.

## Means over 10 seeds

**mask_ratio = 0.3**

| arm | FP16 | Prune33 | Prune50 | P50+I4 |
|---|---|---|---|---|
| Baseline | 93.8% | 69.1% | 35.7% | 31.1% |
| **SAL** | 92.9% | 74.5% | 47.9% | **43.6%** |
| SAL+QAT | 93.0% | 74.8% | 48.7% | 46.2% |
| **DropHead** | **93.7%** | **87.6%** | **75.4%** | **72.4%** |
| No-accum | 93.7% | 84.9% | 65.4% | 60.8% |
| DropHead+QAT | 93.8% | 87.5% | 75.4% | 74.2% |
| No-accum+QAT | 93.8% | 85.0% | 65.8% | 63.9% |

**mask_ratio = 0.5**

| arm | FP16 | Prune33 | Prune50 | P50+I4 |
|---|---|---|---|---|
| Baseline | 93.7% | 69.5% | 35.5% | 30.9% |
| **SAL** | 92.0% | 74.9% | 53.6% | **49.8%** |
| **DropHead** | **92.6%** | **88.4%** | **82.2%** | **80.4%** |
| No-accum | 93.0% | 86.7% | 75.2% | 72.1% |

## Paired, per seed, at P50+I4

| | mask_ratio 0.3 | mask_ratio 0.5 |
|---|---|---|
| DropHead − SAL | **+28.80pp**, sd 4.88, CI [+25.8, +31.8], **10/10** | **+30.61pp**, sd 5.32, CI [+27.3, +33.9], **10/10** |
| No-accum − SAL | +17.27pp, sd 4.33, CI [+14.6, +20.0], 10/10 | +22.29pp, sd 5.20, CI [+19.1, +25.5], 10/10 |

Every seed, both ratios, confidence intervals nowhere near zero.

And DropHead does it **more cheaply**: clean FP16 accuracy 93.72% against SAL's
92.93% at mask_ratio 0.3 (baseline 93.80%). SAL pays almost a point of clean
accuracy; DropHead pays less than a tenth of one. DropHead is also far more
stable — 72.0–73.1% across seeds where SAL spans 34.7–51.2%.

There is no axis in this table on which SAL wins.

## What this falsifies

Cardinal rule 1 in `CLAUDE.md` states that pruned heads accumulating is "the
causal mechanism — the model self-reorganizes to operate without the removed
heads". **The `nonaccum` arm removes only the accumulation and gains +17 to
+22pp.** Accumulation is not the mechanism; it is a liability. Going further to
plain per-step resampling adds another ~10pp on top.

So the defensible version of the project's claim is no longer "progressive
structural damage builds resilience". It is:

> **Training-time random head masking produces very large compression
> resilience** — DropHead reaches 80.4% top-1 at 50% pruning + INT4 where a
> standard fine-tune gets 30.9%, a **+49.5pp** gain at essentially no cost in
> clean accuracy. **The right form is per-step resampling, not progressive
> accumulation.**

That is a better result than the one being claimed, obtained by the simpler
method. It is also not a new method: head dropout has been in the literature
since 2019. The novel part of sal-torch's mechanism is the part that makes it
worse.

## The one interpretation that could rescue accumulation

**SAL is evaluated on a head set it was not trained for.** SAL finishes training
with a specific 30% (or 50%) of heads permanently dead. Evaluation then removes
a *different*, randomly chosen 50%. So the SAL model is scored having lost its
trained-dead heads **plus** an unrelated random half, while DropHead's model
enters evaluation with every head alive and trained to tolerate any subset.

That is the mechanistic explanation for the gap, and it implies a test that has
never been run: **prune exactly the heads SAL killed.** If SAL's adaptation is
real, the aligned case should be its best cell, possibly beating DropHead. If
SAL still loses when the deployed pruned set is its own, accumulation has no
defence left.

This experiment is cheap (it reuses trained checkpoints) but needs a code
change: neither `SALTrainer` nor the prototype records *which* heads were
pruned — `masker_stats` keeps counts only. `HeadMasker` knows them internally.
Until that is run, the honest statement is that **SAL loses badly in the
deployment scenario the README actually documents** (SAL-train, then prune
random heads), and that the aligned scenario is untested.

## Consequences

- **Do not ship `mask_ratio` tuning or the per-architecture ratio table built on
  SAL's accumulating schedule.** The PHASE2B dose-response is real but it is a
  dose-response of the weaker mechanism; DropHead at mask_ratio 0.3 already
  beats SAL at 0.5 by +22.6pp.
- **The +14.5pp headline is not wrong, but it is the wrong comparison.** SAL
  does beat a standard fine-tune by +14.5pp. It loses to a one-line control by
  ~30pp. A README that reports the first without the second is misleading, and
  both numbers now exist.
- **Run the aligned-pruning experiment before any decision about SAL's future.**
  It is the only remaining scenario where accumulation could be the right
  choice, and it is a few GPU-hours.
- If the aligned result does not rescue it, the product's default masking should
  become per-step resampling, and `HeadMasker`'s accumulate-and-hold behaviour
  should be demoted to an option documented as inferior.
