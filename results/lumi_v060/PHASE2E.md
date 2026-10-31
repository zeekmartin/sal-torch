# expF + exp13 — structured selection loses, and DropHead generalises

Two questions, both now answered on 10 and 3 seeds respectively.

1. **expF** — can the draw be made smarter? Two variants against plain DropHead:
   `fi_drophead` (fragility-weighted draw) and `progressive` (ramping ratio).
2. **exp13** — does DropHead's advantage survive a change of architecture and of
   training objective? DINOv2-base (self-supervised ViT) and GPT-2 Medium
   (causal LM, Conv1D attention) against ViT-B/16's supervised setup.

---

## expF — ViT-B/16 / ImageNet-100, 10 seeds, 20 head selections

`fi_drophead` re-scans per-head fragility every 2 epochs and draws fragile heads
at `2x` the base rate, resilient heads at `0.5x`, renormalised so the expected
drop rate matches DropHead's exactly. `progressive` ramps the ratio linearly
from 0.1 to the target, resampled every step, no accumulation.

### mask_ratio = 0.5

| arm | FP16 | P33 | P50 | INT4 | P50+I4 |
|---|---|---|---|---|---|
| baseline | 93.81% ±0.2 | 68.95% ±1.5 | 35.31% ±2.3 | 92.99% ±0.3 | 30.62% ±1.9 |
| **drophead** | 92.67% ±0.1 | **88.39% ±0.2** | **82.24% ±0.2** | 91.90% ±0.3 | **80.39% ±0.3** |
| progressive | 93.25% ±0.2 | 87.25% ±0.3 | 75.54% ±0.4 | 92.59% ±0.2 | 72.35% ±0.5 |
| fi_drophead | 93.09% ±0.2 | 87.49% ±0.3 | 78.04% ±0.5 | 92.54% ±0.3 | 75.62% ±0.7 |

### mask_ratio = 0.3

| arm | FP16 | P33 | P50 | INT4 | P50+I4 |
|---|---|---|---|---|---|
| baseline | 93.79% ±0.2 | 69.79% ±1.4 | 35.65% ±1.8 | 93.02% ±0.3 | 31.06% ±1.3 |
| **drophead** | 93.78% ±0.2 | **87.53% ±0.2** | **75.38% ±0.5** | 93.13% ±0.2 | **72.26% ±0.5** |
| progressive | 93.92% ±0.1 | 85.74% ±0.3 | 67.58% ±0.9 | 93.34% ±0.2 | 63.38% ±1.0 |
| fi_drophead | 93.94% ±0.2 | 86.52% ±0.3 | 70.67% ±1.2 | 93.36% ±0.2 | 67.24% ±1.3 |

### Head-to-head on the key cell (P50+I4), paired per seed

| | mask_ratio 0.3 | mask_ratio 0.5 |
|---|---|---|
| progressive − drophead | **−8.89pp**, CI [−9.32, −8.45], 0/10 | **−8.05pp**, CI [−8.33, −7.76], 0/10 |
| fi_drophead − drophead | **−5.02pp**, CI [−5.64, −4.40], 0/10 | **−4.77pp**, CI [−5.15, −4.39], 0/10 |
| fi_drophead − progressive | +3.86pp, CI [+3.37, +4.36], 10/10 | +3.28pp, CI [+3.00, +3.55], 10/10 |

**Neither variant helps, and both lose on 0 of 10 seeds with tight intervals.**

**Fragility-guided selection is worse than a uniform random draw.** This is a
direct, well-powered confirmation of the project's cardinal rule 1 at a scale it
had not previously been tested: structured selection underperforms random. The
intervals are narrow (±0.4pp), so this is not a question of power. Note that FI
was used here as a *selector*, which is exactly the role the rule forbids — it
remains a diagnostic.

**Ramping the ratio is worse than starting at full ratio.** `progressive` is the
weakest arm at every ratio, which makes the mechanism legible: what matters is
total exposure to the perturbation, and a ramp simply delivers less of it. That
also explains why `fi_drophead` beats `progressive` — it is not that fragility
weighting helps, but that `fi_drophead` keeps DropHead's full-ratio-from-step-0
schedule and only perturbs *which* heads get dropped.

Every arm beats baseline massively (+32 to +50pp on the key cell, 10/10). The
question expF answered is not whether per-step dropout works — Track C settled
that — but whether it can be improved. It cannot, by either of these routes.

---

## exp13 — does it generalise across architectures?

### DINOv2-base / ImageNet-100, 3 seeds, mask_ratio 0.3, 2 epochs

| arm | clean | prune 33% | prune 50% |
|---|---|---|---|
| baseline | 92.70% ±1.7 | 17.18% ±1.5 | 1.55% ±0.2 |
| sal | 88.95% ±2.4 | 39.65% ±3.5 | 2.97% ±0.8 |
| **drophead** | 89.45% ±1.2 | **82.07% ±1.4** | **60.13% ±1.6** |

| comparison | 33% | 50% |
|---|---|---|
| drophead − baseline | **+64.89pp**, 3/3 | **+58.58pp**, 3/3 |
| drophead − sal | **+42.42pp**, 3/3 | **+57.16pp**, 3/3 |
| sal − baseline | +22.47pp, 3/3 | +1.42pp, 3/3 |

This replicates Track C on a **second architecture and a second training
objective** — DINOv2 is self-supervised, ViT-B/16 supervised. DropHead's clean
accuracy cost is the same as SAL's (−3.25pp vs −3.75pp), so the advantage is not
bought with clean accuracy. As in exp12, SAL scored on *random* sets still beats
baseline (+22.5pp at 33%): the adaptation exists, it is just specific to the set
SAL trained for.

### GPT-2 Medium / SST-2, 3 seeds, mask_ratio 0.3, 2 epochs

| arm | clean | prune 33% | prune 50% |
|---|---|---|---|
| baseline | 92.74% ±0.6 | 89.42% ±0.4 | 84.06% ±1.9 |
| sal | 92.47% ±1.2 | 90.00% ±0.9 | 85.42% ±0.8 |
| **drophead** | 92.09% ±0.6 | **90.74% ±0.2** | **88.39% ±0.4** |

| comparison | 33% | 50% |
|---|---|---|
| drophead − baseline | +1.32pp, 3/3 | **+4.33pp**, 3/3 |
| drophead − sal | +0.74pp, 2/3 | **+2.97pp**, 3/3 |
| sal − baseline | +0.58pp, 2/3 | +1.36pp, 3/3 |

The ordering is the same — drophead > sal > baseline — but the margins are an
order of magnitude smaller. The reason is visible in the baseline row: **GPT-2 on
SST-2 is already robust**, retaining 84.1% at 50% head pruning, where DINOv2's
baseline collapses to 1.55%. There is almost no headroom to win. This supports
the task-difficulty reading already recorded in `CLAUDE.md`: SAL-family methods
pay off where compression actually hurts, and a binary sentiment task on a 24-layer
model is not such a case. It also means **GPT-2/SST-2 is a poor benchmark for
this family of methods**, whichever direction it points.

A methodological caution about these GPT-2 numbers: the first attempt produced
`losses [nan, nan]` on all three arms and 49.08% on every cell — 428/872, the
SST-2 majority class. Under ROCm's default SDPA path GPT-2 returns finite logits
and corrupted gradients (grad norms of 7.7e13, 3.6e18, `inf` within 40 steps,
against 15-140 under eager), so `clip_grad_norm_` scaled by `1/inf = 0` and the
model never updated. The numbers above come from `attn_implementation="eager"`,
with losses falling 0.35 → 0.15 and clean accuracy where SST-2 fine-tuning should
land. Because only the backward pass was affected, a forward-only check of the
masker passed — and did: masking moves GPT-2 logits by 4.2 with every head off.

---

## Where this leaves the method

Three independent lines now agree:

1. **Per-step resampled head dropout is the strong, general method.** It wins on
   ViT-B/16 (Track C), DINOv2 (exp13), and GPT-2 (exp13), at every ratio tested.
2. **It cannot be improved by choosing heads cleverly** (expF). Uniform random
   beats fragility-weighted, 0/10 seeds.
3. **SAL's accumulate-and-hold schedule wins only on its own trained set**
   (exp12, +2.11pp over DropHead at mask_ratio 0.5, 10/10) — and that set is not
   uniform per layer, so `slice_heads()` cannot express it and the gain is not
   currently deployable as size or speed.

The open gating experiment is unchanged and now more clearly the decisive one:
**make `HeadMasker` draw uniformly per layer and re-run exp12.** If SAL keeps its
+2.1pp under a constrained draw, there is a deployable product claim. If it does
not, DropHead is the product and SAL's schedule is a research result.
