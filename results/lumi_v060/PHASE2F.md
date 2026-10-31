# exp14 — SAL's advantage survives the uniform constraint, and is now sliceable

`PHASE2D.md` left one gating question. SAL beat DropHead by +2.11pp on its own
trained head set, but the set was not uniform per layer, so `slice_heads()` could
not express it: the accuracy win existed only under masking, which removes no
weight and no matmul. The open question was whether constraining the draw to an
equal quota per layer — the thing that makes the set sliceable — would cost the
adaptation, since it constrains what SAL may choose.

**It does not.** And the gain now survives physical slicing.

Same protocol as exp12: ViT-B/16 / ImageNet-100, 10 seeds, 5 epochs, every arm
scored under two perturbations of identical size — SAL's own trained set
(`aligned`) and random sets of the same size averaged over 20 draws shared across
arms (`random`). New in exp14: `SALConfig(uniform_per_layer=True)`, the random
draws are uniform per layer too (the shapes must match or the comparison measures
the shape), and both sets are additionally applied with `slice_heads()`.

## mask_ratio = 0.5 (72 heads, 6 per layer)

| arm | clean | aligned | random | A−R | sliced aligned | sliced random |
|---|---|---|---|---|---|---|
| baseline | 93.77% | 31.56% | 36.70% | −5.14 | 31.55% | 34.92% |
| **sal** | 91.32% | **83.62%** | 54.56% | **+29.06** | **83.61%** | 53.33% |
| drophead | 92.36% | 81.60% | 82.16% | −0.57 | 81.59% | 82.25% |

Sliced model: **85.9M → 71.7M parameters, 16.5% smaller.**

## mask_ratio = 0.3 → effective 0.333 (48 heads, 4 per layer)

| arm | clean | aligned | random | A−R | sliced aligned | sliced random |
|---|---|---|---|---|---|---|
| baseline | 93.86% | 67.27% | 70.36% | −3.09 | 67.27% | 69.37% |
| **sal** | 92.44% | **89.05%** | 75.38% | **+13.67** | **89.05%** | 74.64% |
| drophead | 93.51% | 87.57% | 87.98% | −0.41 | 87.57% | 87.91% |

Sliced model: **85.9M → 76.4M parameters, 11.0% smaller.** The requested 0.3 is
reported as 0.333 because an equal quota needs a target divisible by the layer
count: 0.3 of a 12×12 model is 43 heads, which no constraint on the draw can
split evenly, so the target snaps to 48.

## The comparisons that decide it (paired per seed)

| | mask_ratio 0.3 | mask_ratio 0.5 |
|---|---|---|
| SAL: aligned vs random | **+13.67pp**, CI [+11.46, +15.89], 10/10 | **+29.06pp**, CI [+26.78, +31.33], 10/10 |
| **baseline: aligned vs random** | −3.09pp, CI [−8.89, **+2.70**], 4/10 | −5.14pp, CI [−12.79, **+2.51**], 2/10 |
| SAL-aligned vs DropHead-random | **+1.08pp**, CI [+0.31, +1.84], 8/10 | **+1.45pp**, CI [+0.84, +2.07], 9/10 |
| SAL-aligned vs DropHead-**aligned** | **+1.48pp**, CI [+1.11, +1.85], 10/10 | **+2.02pp**, CI [+1.54, +2.50], 10/10 |
| **sliced** SAL-aligned vs **sliced** DropHead-random | **+1.15pp**, CI [+0.35, +1.95], 8/10 | **+1.36pp**, CI [+0.53, +2.18], 8/10 |
| sliced vs masked, SAL aligned | +0.00pp, CI [−0.01, +0.01] | −0.01pp, CI [−0.02, +0.01] |

**1. The constraint is free.** Under the global draw exp12 measured +10.92pp and
+31.05pp for aligned-vs-random; under the uniform draw, +13.67pp and +29.06pp.
Within noise of each other at 0.5 and better at 0.3. Constraining *which* heads
SAL may pick does not cost the adaptation — only the quota per layer is fixed,
and selection within a layer stays uniformly random.

**2. It is still adaptation, not selection.** The `baseline` control behaves
exactly as in exp12: its aligned-vs-random interval spans zero at both ratios and
the point estimate is *negative* (−3.09, −5.14pp), on 4/10 and 2/10 seeds. An
untrained model does not find SAL's set easy. The set is not made of unimportant
heads; the trained model's relationship to it is what matters.

**3. The gain survives physical slicing, which is the new result.** Masked and
sliced accuracy agree to **±0.01pp over 10 seeds** — the equivalence is exact, as
it should be, and now measured rather than assumed. So the comparison can be
restated in deployment terms: at **16.5% fewer parameters**, a SAL model pruned
on its own set scores **+1.36pp** above a DropHead model pruned on a random set
of the same size, 8/10 seeds. The accuracy advantage is a real, smaller model.

**4. The cleanest same-set comparison is the largest.** Scored on the *identical*
head set, SAL beats DropHead by **+2.02pp on 10/10 seeds** at mask_ratio 0.5
(+1.48pp at 0.333). This removes the last asymmetry: neither arm gets a set the
other did not face.

## What is still true against SAL

- **DropHead remains far better when the deployed set is unknown.** SAL on a
  random set scores 54.56% against DropHead's 82.16% at mask_ratio 0.5. That
  27.6pp gap is the price of specialisation and it has not moved.
- **SAL costs ~1pp of clean accuracy** (91.32% vs DropHead's 92.36% and
  baseline's 93.77% at mask_ratio 0.5), so the net advantage over DropHead at
  deployment is ~1.4pp, not ~2pp.
- **The margin is small in absolute terms.** +1.36pp sliced, with a confidence
  interval reaching down to +0.53pp. It is consistent and seed-stable, not large.

## The product decision this enables

The conditional claim from `PHASE2D.md` is now unconditional, and the condition
that blocked it is gone:

> For a deployment that prunes a **known** head set, train with
> `SALConfig(uniform_per_layer=True)`, save `pruned_heads()` with the
> checkpoint, and slice exactly that set. The result is a model 16.5% smaller
> than the original that scores +1.4pp above the best alternative at the same
> size, on 8-10 of 10 seeds.
>
> For a deployment where the pruned set is **not** under your control, use
> per-step resampled head dropout. It is better there by ~28pp.

Both statements are needed. A page quoting either alone misrepresents the method.

## Corrections to earlier phases

- `PHASE2D.md`'s catch — "the accuracy win cannot presently become a size or
  speed win" — is **resolved**, not merely restated. It was accurate when
  written; `uniform_per_layer` removes it.
- `PHASE2D.md` recommended making the draw uniform and re-running exp12 as "the
  gating experiment for whether SAL's advantage is deployable at all". That
  experiment is this one, and the answer is yes.
- A size reduction is not yet a measured *speed* reduction. Attention is roughly
  a third of a ViT block, and v0.5.1 measured 1.1x GPU / 1.2x CPU for 50% head
  slicing. The joint head+MLP path in `sal-torch-pro` is where the 2x target
  lives; nothing here changes that.

## Open items

- The `prune_fraction` default is still 0.33. The dose-response result
  (`PHASE2B.md`) and this table both favour matching the training ratio to the
  deployment ratio, so a 50% deployment target should train at 0.5.
- Whether `uniform_per_layer` should become the default. It is strictly more
  deployable and costs nothing measured here, but it changes the shipped
  mechanism, so it is a product decision rather than a measurement.
- Nothing here was tested outside ViT-B/16. exp13 showed the DropHead ordering
  holds on DINOv2 and GPT-2, but the aligned/sliced path has only been measured
  on one architecture.
