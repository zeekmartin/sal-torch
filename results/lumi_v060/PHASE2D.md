# exp12 — aligned pruning rescues accumulation, with one condition

Track C concluded that plain DropHead beats SAL by ~30pp and that accumulation
is "not the mechanism, it is a liability" (`PHASE2C.md`). It also flagged the one
interpretation that could overturn that: **SAL was being scored on a head set it
was never trained for.** SAL finishes training with a specific set dead, and the
evaluation removed a *different* random set, so SAL lost its trained-dead heads
*plus* an unrelated half while DropHead entered evaluation with every head alive.

exp12 ran that test. The caveat was right, and Track C's headline reading is
wrong.

ViT-B/16 / ImageNet-100, 10 seeds, every arm scored under two perturbations of
**identical size**: SAL's own killed set (`aligned`), and random sets of the same
count averaged over 20 draws shared across arms (`random`). Both applied by
masking, so the two cells differ only in *which* heads are gone.

## mask_ratio = 0.5

| arm | clean | aligned | random | aligned − random |
|---|---|---|---|---|
| baseline | 93.82% | 38.65% | 35.90% | +2.74 |
| **sal** | 91.92% | **83.91%** | 52.86% | **+31.05** |
| drophead | 92.57% | 81.63% | 81.80% | −0.17 |

## mask_ratio = 0.3

| arm | clean | aligned | random | aligned − random |
|---|---|---|---|---|
| baseline | 93.70% | 71.96% | 74.97% | −3.02 |
| **sal** | 92.95% | **89.65%** | 78.73% | **+10.92** |
| drophead | 93.72% | 88.22% | 88.65% | −0.43 |

## The three comparisons that decide it

| | mask_ratio 0.3 | mask_ratio 0.5 |
|---|---|---|
| SAL: aligned vs random | **+10.92pp**, CI [+8.5, +13.3], 10/10 | **+31.05pp**, CI [+27.5, +34.6], 10/10 |
| **baseline: aligned vs random** | −3.02pp, CI [−9.3, +3.3], 4/10 | +2.74pp, CI [−2.1, +7.6], 8/10 |
| SAL-aligned vs DropHead-random | **+1.00pp**, CI [+0.3, +1.7], 8/10 | **+2.11pp**, CI [+1.4, +2.8], 10/10 |

**1. The adaptation is real.** SAL recovers +31pp simply by being asked to lose
the heads it trained without, on 10/10 seeds, confidence interval far from zero.

**2. It is adaptation, not selection.** This is what the `baseline` row settles,
and it is the control no previous run had. If SAL's pruned set were merely a set
of unimportant heads, an *untrained* model would also lose little from it. It
does not: baseline on the aligned set is statistically indistinguishable from
baseline on a random set (CI spans zero at both ratios, and the point estimate is
*negative* at 0.3). The set is not special in itself — the trained model's
relationship to it is.

**3. DropHead is uniformly robust, and SAL's specialisation narrowly beats it.**
DropHead scores the same on the aligned set as on random ones (−0.17, −0.43pp):
exactly what per-step resampling should produce, with no special relationship to
any particular set. SAL, scored on its own set, beats DropHead's general
robustness by **+2.11pp at mask_ratio 0.5 on 10/10 seeds** (+1.00pp at 0.3). The
edge grows with the ratio. With INT4 on top, SAL-aligned 81.27% against
DropHead-random 79.89%.

## What this means for the product, and the catch

The honest statement is now **conditional**, and the condition is not currently
satisfiable:

> SAL's accumulate-and-hold schedule produces a model that is *specialised* to
> one head set. On that set it is the best option measured — +2.1pp over
> DropHead. On any other set of the same size it is far worse than DropHead
> (52.9% against 81.8%). DropHead buys general robustness; SAL buys a higher
> peak at one point.

**The catch: SAL's set cannot be sliced.** `HeadMasker` draws heads from the
whole model, so the set is not uniform per layer — measured
`[8,4,5,7,3,7,4,7,5,8,7,7]` at mask_ratio 0.5 — while `slice_heads()` requires
the same count in every layer, because architectures store a single head count.
So SAL's +2.1pp is reachable today only by *masking*, which keeps every weight
and every matmul: **the accuracy win cannot presently become a size or speed
win.** DropHead has no such problem, since any uniform set serves it equally.

That makes the fix concrete and small: **make `HeadMasker` draw uniformly per
layer.** Then the set SAL trains for is exactly a `slice_heads()` argument, and
+2.1pp at 50% pruning becomes deployable. Whether uniform drawing costs any of
the adaptation is an open question — it constrains the draw, so it might.

## Corrections to PHASE2C

- "There is no axis in this table on which SAL wins" — **wrong** as a general
  statement. It was true of that table, which scored SAL only on random sets.
  On its own set SAL wins both against its random-set self and against DropHead.
- "Accumulation is not the mechanism, it is a liability" — **withdrawn.**
  Accumulation produces a real, large, seed-stable adaptation. It is a liability
  only when the deployed set is not the trained set.
- The +49.5pp DropHead-over-baseline figure stands, as does DropHead's lower
  clean-accuracy cost. What changes is that it is no longer the best option
  available, only the best *unconditional* one.

## Recommended direction

1. **Make `HeadMasker` prune uniformly per layer** and re-run exp12. This is the
   gating experiment for whether SAL's advantage is deployable at all.
2. **Ship DropHead as the default** for users who cannot control which heads get
   removed — the README's documented workflow. It is better there by ~30pp.
3. **Keep accumulation as the recommended path for a known deployment target**,
   and document that the pruned set must be saved and reused. `pruned_heads()`
   now exists for exactly this; without it the advantage is unrecoverable.
4. Report both numbers. A page that quotes either alone misrepresents the method.
