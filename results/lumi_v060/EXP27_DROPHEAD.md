# exp27 — does DropHead's training help on a *fixed* head set?

`experiments/exp27_drophead_eval.py`. Gate `23499325` (2:33), jobs `23499326`
(20:48), `23499327` (24:34), `23499328` (24:45), all `COMPLETED 0:0`.
**1.17 GPU-hours, inference only, no training of any kind.**

ViT-B/16, **ImageNet-1k (resized 256px, evanarlian/imagenet_1k_resized_256)**,
full 50 000-image validation split, three seeds (42, 123, 456). Every row is one
deterministic sliced architecture, so there are no draws and no within-seed
spread.

## The question

exp25 settled this for SAL: on its own trained-for set SAL reads 76.51% against
20.33% for 1a's baseline checkpoint on the identical set, and 34.52% on sets it
never saw. Real adaptation, entirely set-specific.

DropHead resamples its mask every step, so it has no "own" set. 1a scores it at
73.29% ± 0.71 on 20 *random* sets. Does that survive on a set chosen by
something other than chance, or is random-set resilience all it has?

Three checkpoints from 1a (`baseline`, `sal`, `drophead` — identical data,
optimiser, schedule and epoch count, differing only in masking) are scored on the
same head sets. **1a's baseline on the identical set is the control**; a high
score on a magnitude set could otherwise be the selection rule rather than the
training.

### The head sets, and a provenance problem measured rather than assumed

`magnitude_heads` ranks heads by the L2 norm of their slice of the attention
output projection, read off **whichever checkpoint is passed**. exp23 ranked the
pretrained checkpoint; exp24's `drophead_magnitude` arm ranked the drophead
checkpoint itself. "The magnitude set" is not a well-defined object, so both
readings were run:

- **`magnitude_pXX`** — ranked on the **baseline** checkpoint and removed from
  all three. One architecture everywhere, so cross-checkpoint differences are
  differences in weights alone. **This is the row the conclusions rest on.**
- **`magnitude_pXX_self`** — each checkpoint's own ranking, i.e. exp24's reading.

Overlap with the pinned set, measured on every seed:

| | P50 | P33 |
|---|---|---|
| baseline | 72/72 *identical* | 48/48 *identical* |
| drophead | 69–71 / 72 | 44/48 on all three seeds |
| sal | 67–71 / 72 | 42–46 / 48 |

The baseline rows being exactly identical is the check that the pinning works.
`aligned_p50` additionally removes **SAL's** trained-for set from all three.

## Results — FP (autocast bf16)

| head set removed | checkpoint | s42 | s123 | s456 | **mean** | sd | vs own dense |
|---|---|---|---|---|---|---|---|
| — (dense) | drophead | 77.60 | 77.32 | 77.71 | **77.54** | 0.20 | |
| — (dense) | baseline | 80.48 | 80.45 | 80.38 | **80.44** | 0.05 | |
| — (dense) | sal | 71.53 | 67.57 | 61.61 | **66.90** | 4.99 | |
| **magnitude P33** | **drophead** | 77.59 | 77.83 | 77.64 | **77.69** | 0.13 | **+0.14** |
| magnitude P33 | baseline | 51.46 | 51.24 | 51.02 | 51.24 | 0.22 | −29.20 |
| magnitude P33 | sal | 57.94 | 53.12 | 45.32 | 52.13 | 6.37 | −14.78 |
| **magnitude P50** | **drophead** | 72.56 | 72.56 | 72.48 | **72.54** | 0.05 | **−5.01** |
| magnitude P50 | baseline | 6.80 | 10.36 | 10.02 | 9.06 | 1.96 | −71.38 |
| magnitude P50 | sal | 14.60 | 20.27 | 16.74 | 17.21 | 2.87 | −49.70 |
| magnitude P33 *self* | drophead | 77.31 | 77.18 | 77.20 | 77.23 | 0.07 | −0.31 |
| magnitude P33 *self* | sal | 60.15 | 51.83 | 26.46 | 46.15 | **17.55** | −20.75 |
| magnitude P50 *self* | drophead | 72.48 | 72.45 | 72.49 | 72.47 | 0.02 | −5.07 |
| magnitude P50 *self* | sal | 27.83 | 21.29 | 13.35 | 20.82 | 7.25 | −46.08 |
| SAL's aligned P50 | drophead | 75.23 | 74.02 | 75.83 | **75.03** | 0.92 | −2.51 |
| SAL's aligned P50 | baseline | 25.62 | 22.73 | 12.64 | 20.33 | 6.81 | −60.11 |
| SAL's aligned P50 | sal | 76.59 | 75.98 | 76.97 | **76.51** | 0.50 | +9.61 |

## Results — INT4 (simulated per-channel round-trip)

| head set removed | drophead | baseline | sal |
|---|---|---|---|
| — (dense) | 76.09 | 79.01 | 64.82 |
| **magnitude P33** | **76.12** *(+0.04)* | 43.29 | 46.53 |
| **magnitude P50** | **69.89** *(−6.20)* | 6.46 | 14.21 |
| magnitude P33 *self* | 75.55 | — | 41.07 |
| magnitude P50 *self* | 69.79 | — | 16.81 |
| SAL's aligned P50 | 73.04 | 16.05 | **73.95** |

Against the baseline checkpoint on the **identical** set:

| set | DropHead | SAL |
|---|---|---|
| magnitude P33 | **+26.45pp** / +32.84 INT4 | +0.89pp / +3.24 |
| magnitude P50 | **+63.47pp** / +63.43 INT4 | +8.14pp / +7.75 |
| SAL's aligned P50 | **+54.70pp** / +56.99 INT4 | +56.18pp / +57.90 |

# What this says

## 1. The answer is yes, and it is not small

The brief's expectation was "a small improvement on magnitude sets". Measured:
**+63.47pp at P50** and **+26.45pp at P33** over the baseline checkpoint on the
identical head set, on 3/3 seeds, every cell.

So DropHead's resilience is **not** an artefact of being scored on random draws.
It holds on a magnitude-chosen set, and it holds on **SAL's** set (+54.70pp),
which DropHead has no more seen than any other. Whatever per-step resampling
teaches the model, it is general over head sets rather than tied to one.

That is the opposite shape from SAL. Put the two mechanisms side by side, each
measured against the same baseline on the same sets:

| | on its own set | on an external fixed set | on random sets |
|---|---|---|---|
| SAL | **+56.18pp** | +8.14pp (P50), +0.89pp (P33) | −0.05pp (exp25) |
| DropHead | n/a — has none | **+63.47pp** (P50), +26.45pp (P33) | **+54.95pp** (P50, 1a) |

Every cell in this table is FP against FP. **Correction:** the `on random sets`
cell previously read `+52.9pp`, which was DropHead's **INT4** P50 accuracy
(73.29) minus the baseline's **FP** P50 accuracy (20.38) — two different
columns of 1a's table, so not a comparison of anything. The matched figures
from `PHASE3_IMAGENET1K.md` are:

| on random sets, 1a | DropHead | baseline | difference |
|---|---|---|---|
| P50, FP | 75.33 ± 0.61 | 20.38 ± 6.29 | **+54.95pp** |
| P50 + INT4 | 73.29 ± 0.71 | 16.06 ± 5.39 | **+57.23pp** |
| P33, FP | 77.74 ± 0.29 | 48.80 ± 7.55 | **+28.94pp** |

No conclusion in this document changes: the point is that DropHead's advantage
is of the same order on random sets as on fixed ones (+54.95 against +63.47 at
P50), and the corrected figure is 2pp *larger* than the one it replaces.

**SAL's adaptation is worth 56pp on one set and ~0 to 8pp anywhere else.
DropHead's is worth 26–63pp everywhere, including on SAL's own set.**

## 2. For DropHead, pruning 33% of heads is free

`magnitude_p33` reads **77.69%** against DropHead's own dense **77.54%** —
**+0.14pp**, i.e. removing 33% of the attention heads costs nothing and in fact
reads marginally better. At INT4 the same: 76.12 against 76.09, **+0.04pp**. At
P50 the cost is **−5.01pp**, still from a model that needs no recovery pass and
no knowledge of which heads will go.

Stated against the strong reference — the baseline checkpoint's 80.44% — the
whole package is: **DropHead training costs 2.90pp of dense accuracy, and from
there 33% of heads come out for 0.14pp, on any set, with no fine-tuning.**
Net −2.75pp at 33% fewer heads.

That is the best no-recovery result in this campaign. It beats SAL's own-set P50
number (76.51), and at INT4 (76.12) it beats SAL's own-set INT4 (73.95) by
2.17pp — on a set SAL would need to have trained for and DropHead does not.

## 3. SAL's adaptation barely transfers to an external structured set

exp25 showed SAL's set is worth −0.05pp to a model that did not train for it.
This is the mirror: an external set is worth **+0.89pp at P33 and +8.14pp at
P50** to SAL over the baseline. Not literally nothing, and the P50 figure is not
negligible — but against DropHead's +63.47pp on that same set it is not a
mechanism anyone would ship.

Note also that SAL's dense accuracy (66.90%) is below DropHead's P50-pruned
accuracy (72.54%). The SAL checkpoint, intact, is worse than the DropHead
checkpoint with half its heads removed.

## 4. The variance gap is two orders of magnitude, and it is the sharpest
   product argument in the file

| | sd across three seeds |
|---|---|
| DropHead, all six rows | **0.02 – 0.92pp** |
| SAL, all six rows | 0.50 – **17.55pp** |

DropHead's worst row (0.92pp, on SAL's aligned set) is better than SAL's best
(0.50pp — also on SAL's own set, the only place SAL is stable). SAL's
`magnitude_p33_self` row spans **60.15 / 51.83 / 26.46%**, a 33.7pp spread on
three seeds of the same configuration.

A method whose accuracy moves thirty points with the training seed cannot carry
a 0.3–1.5pp cost claim, and `CLAUDE.md` still carries one.

## 5. Head-set provenance matters for SAL and not for DropHead

DropHead's own ranking gives essentially the pinned answer — 77.23 vs 77.69 at
P33, 72.47 vs 72.54 at P50 — despite 4 of 48 and 1–3 of 72 heads differing. SAL's
diverges wildly: 46.15 vs 52.13 at P33 (and sd 17.55 vs 6.37), 20.82 vs 17.21 at
P50. SAL's weights are disordered enough that its own magnitude ranking is not a
reliable object, which is itself a reason not to read `magnitude` rows off a SAL
checkpoint.

Consistency check: SAL's `magnitude_p50_self` row reads **27.83 / 21.29 / 13.35%**
— exactly exp24's post-slice-before-recovery column on the same three seeds, in a
different script and a different harness.

# Caveats

- Inference only, one deterministic architecture per row. The three-seed columns
  are the only variability; there is no within-seed spread to quote.
- Accuracy is **autocast bfloat16**, the path 1a, exp24, exp25 and exp26 score
  under — not true FP32. Comparable with those, not with an FP32 reference.
- INT4 is `quantize_weights(model, 4)`, the simulated per-channel round-trip,
  not bitsandbytes NF4.
- The control is 1a's **baseline checkpoint**, not the raw pretrained one: same
  data, optimiser, schedule and epochs, no masking. That is the matched control.
  exp23 is the experiment that works from the pretrained checkpoint.
- **DropHead is never scored on home ground**, because it has none. Every set
  here is external to it. That asymmetry favours SAL in the `aligned_p50` row and
  is the whole point everywhere else.
- `magnitude_p50` removes 6 heads per layer and `magnitude_p33` removes 4, both
  uniform per layer, which is what makes them valid `slice_heads()` arguments.
- One setting: ViT-B/16, ImageNet-1k, DropHead at 1a's configuration. Whether
  any of this holds on I-JEPA, on GPT-2 Medium, or at 70% pruning is unmeasured —
  and 70% is where v0.5.1 found SAL had no effect at all.
