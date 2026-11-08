# exp25 — the 1a checkpoints scored on SAL's own head set

`experiments/exp25_aligned_eval.py`. Gate `23486722` (5:29), jobs `23486725`,
`23486726`, `23486727`, all `COMPLETED 0:0` in ~13 min each. **1.1 GPU-hours,
inference only, no training of any kind.**

ViT-B/16, **ImageNet-1k (resized 256px, evanarlian/imagenet_1k_resized_256)**,
full 50 000-image validation split, three seeds (42, 123, 456). Every row is one
deterministic architecture, so there are no draws and no within-seed spread.

## Why this measurement was missing

1a scores its pruned cells as the mean over **20 random head selections**. That
is the right protocol for a model that was not trained for any particular set —
and it is the wrong one for SAL, which trained for exactly one. SAL spent
thousands of steps with a specific 72 of 144 heads masked; 1a then evaluated it
on 20 sets, none of them that one. exp24 ran into the consequence by accident:
its intermediate "accuracy after slicing, before recovery" column read ~76.5%
where 1a's table said 34.52%.

This measures it directly, with the control the claim requires: **the same 72
heads removed from 1a's baseline checkpoint**, which saw identical data,
optimiser and schedule with no masking.

## Results — three seeds, full validation split

| row | checkpoint | s42 | s123 | s456 | **mean** | mean INT4 |
|---|---|---|---|---|---|---|
| dense | SAL | 71.53 | 67.57 | 61.61 | **66.90** | 64.82 |
| dense | baseline | 80.48 | 80.45 | 80.38 | **80.44** | 79.01 |
| aligned P33 | SAL | 72.70 | 71.51 | 74.06 | **72.76** | 70.28 |
| aligned P33 | baseline | 52.84 | 53.37 | 43.50 | **49.90** | 44.41 |
| **aligned P50** | **SAL** | **76.59** | **75.98** | **76.97** | **76.51** | **73.95** |
| **aligned P50** | **baseline** | 25.62 | 22.73 | 12.64 | **20.33** | 16.05 |

Against each checkpoint's own dense accuracy:

| | SAL | baseline |
|---|---|---|
| aligned P33 | **+5.85pp** | −30.54pp |
| aligned P50 | **+9.61pp** | −60.11pp |
| aligned P50 + INT4 | **+9.13pp** | −62.96pp |

**SAL minus baseline, on the identical head set: +22.85pp at P33 and +56.18pp at
P50 (+57.90pp with INT4).** On 3/3 seeds, every cell, no exceptions.

# What this says

## 1. The adaptation is real, and it is entirely set-specific

Put the new numbers beside 1a's random-draw columns:

| at P50 | on **20 random** sets (1a) | on **SAL's own** set (exp25) | difference |
|---|---|---|---|
| SAL | 34.52 ± 9.49 | **76.51** | **+41.99pp** |
| baseline | 20.38 ± 6.29 | 20.33 | **−0.05pp** |

Read the baseline row first, because it is the one that settles the question.
**For a model that did not train for it, SAL's set is worth exactly what a random
set is worth: 20.33 against 20.38, a difference of 0.05pp.** The set is not
intrinsically a good set of heads to remove — it is not a magnitude ranking, not
a fragility ranking, not anything that transfers. It was drawn at random.

And SAL on that same set scores 76.51 against 34.52 on sets it never saw. So
what SAL learned is **not** "operate with fewer heads". It is "operate without
*these* heads", and that knowledge is worth 42pp on the set in question and
nothing at all off it.

This is the first result in the campaign where SAL's mechanism produces an
effect that no control reproduces. `PHASE3_IMAGENET1K.md` §2 showed DropHead
beating SAL by 42.66pp at P50+INT4 on random sets; that remains true and is
still the right comparison for prune-anywhere resilience. It is simply not a
measurement of whether the accumulation mechanism does anything, and this one is.

## 2. SAL's "clean accuracy cost" is an artefact of scoring the wrong architecture

SAL's dense accuracy is 66.90% against the baseline's 80.44% — a 13.54pp cost,
and the reason `CLAUDE.md` carries a "costs 0.3–1.5pp of clean accuracy" claim
that this setting does not support. But the dense architecture is **not what SAL
produces**. Remove the heads it trained without and it reads 76.51%, which is
9.61pp *better* than its own dense score.

Carrying those 72 heads is actively harmful: they are heads whose function was
redistributed elsewhere, still contributing to the residual stream. The honest
statement of SAL's cost on ImageNet-1k is therefore **80.44 → 76.51, i.e.
−3.93pp at a 16.5% smaller model**, not −13.54pp. Every table that quotes SAL's
dense number as its clean accuracy, including 1a's and this campaign's own
summaries, overstates the cost by 9.6pp.

## 3. Where it leaves SAL, next to the alternatives

At P50 + INT4, the number the paper turns on:

| method | P50+INT4 | scored on | needs post-prune training |
|---|---|---|---|
| SAL, random sets (1a) | 30.63 | 20 unseen sets | no |
| DropHead, random sets (1a) | 73.29 ± 0.71 | 20 unseen sets | no |
| **SAL, its own set (exp25)** | **73.95** | **1 trained-for set** | **no** |
| magnitude + recovery (exp23) | 75.06 | 1 fine-tuned-for set | **yes, 5 epochs** |
| SAL → aligned + recovery (exp24) | 75.66 | 1 fine-tuned-for set | yes, 5 epochs |

SAL on its own set is **+0.66pp over DropHead** and **−1.11pp under magnitude +
recovery** — and unlike the latter it needs no fine-tuning pass after pruning.
That is a defensible position, with two conditions attached that are not small:

- The deployment head set must be **fixed at training time**. SAL cannot be
  pruned anywhere else: 34.52% on an unseen set is the same model.
- The ratio must be known too. At P33 — a subset of SAL's own set, so an
  architecture it never trained for — it reads 72.76%, below the 78.85% that
  magnitude + recovery gives.

DropHead buys 73.29% on *any* set at ±0.71 and keeps 77.54% dense. For a product
whose promise is compression resilience, that is still the stronger offer. SAL's
offer is different and narrower: one known architecture, no recovery pass, 76.5%.

## 4. The P33 rows carry a one-head confound

`p33_subsets_identical: false` on all three seeds. The P33 row is the
lowest-magnitude 48 of SAL's 72 heads, and the magnitudes are read off whichever
checkpoint is being sliced, so the SAL and baseline subsets differ — by **1 head
of 48** on seed 42 and **2 of 48** on seeds 123 and 456. The P33 comparison is
therefore not strictly the same architecture, though it is within one or two
heads of it and the gap it reports is 22.85pp.

Both head sets are recorded in each JSON. Re-running with the subset pinned to
SAL's ranking costs ~1.1 GPU-hours and has not been done, because no conclusion
above turns on the P33 row. The P50 rows, which carry everything, are exact:
there the quota equals SAL's set and `aligned_is_subset` is `false`.

# Caveats

- Inference only, one deterministic architecture per row, so there is no spread
  within a seed to quote. The three-seed columns are the only variability.
- The accuracy column is **autocast bfloat16**, the path 1a and exp24 score
  under — not true FP32. It is comparable with those experiments, not with an
  FP32 reference.
- INT4 is the simulated per-channel round-trip, `quantize_weights(model, 4)`,
  not bitsandbytes NF4.
- The control is 1a's **baseline checkpoint**, not the raw pretrained one: same
  data, optimiser, schedule and epoch count, no masking. That is the matched
  control. The pretrained checkpoint answers a different question and is exp23's
  subject.
- One setting: ViT-B/16, ImageNet-1k, `--mask-ratio 0.5 --uniform-per-layer`.
  Whether the aligned-set advantage survives on I-JEPA or at 70% is unmeasured.
