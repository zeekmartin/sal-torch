# exp15 — on I-JEPA, part of SAL's advantage is selection, and the rest reproduces

`PHASE2F.md` established on ViT-B/16 that SAL's advantage on its own trained head
set is **adaptation, not selection**: a standard fine-tune scored on that same set
did *worse* than on a random set of equal size (−5.14pp at mask_ratio 0.5), so the
set is not made of unimportant heads. That claim was, and remains, scoped to
ViT-B/16 / ImageNet-100.

exp15 runs the same design on a second architecture. **The control does not
behave the same way, and the reason matters for how the result should be sold.**

## Protocol

I-JEPA ViT-H/14 (`facebook/ijepa_vith14_1k`), ImageNet-100, 5 epochs, batch 16,
8192 training images, linear probe on 5000/2500, **3 seeds** (42, 123, 456),
`mask_ratio 0.5`, `uniform_per_layer=True`. 32 layers × 16 heads = 512 heads, of
which **256 pruned, exactly 8 per layer**, effective fraction 0.500. Every arm is
scored on SAL's trained set (`aligned`) and on **10 random sets of identical
shape** shared across arms (`random`), each additionally applied through
`slice_heads()` (3 draws) and through simulated INT4. AMD MI250X, ROCm 6.2,
transformers 4.47.1. 3.2 h per seed, ~9.6 GPU-h total.

## Per-seed cells

| seed | arm | clean | aligned | random | A−R | sliced aln | sliced rnd | clean I4 | aln I4 | rnd I4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | control | 83.44% | 68.56% | 62.50% | +6.06 | 68.72% | 67.63% | 83.04% | 65.68% | 62.15% |
| 42 | **sal** | 83.32% | **72.20%** | 63.94% | **+8.26** | **72.24%** | 68.81% | 83.48% | **68.76%** | 63.66% |
| 42 | drophead | 82.88% | 69.52% | 63.42% | +6.10 | 69.52% | 68.25% | 83.08% | 65.96% | 63.30% |
| 123 | control | 82.56% | 61.20% | 58.45% | +2.75 | 61.20% | 56.05% | 82.60% | 58.20% | 51.69% |
| 123 | **sal** | 82.80% | **64.44%** | 59.71% | **+4.73** | **64.44%** | 56.99% | 83.00% | **62.00%** | 52.78% |
| 123 | drophead | 81.92% | 61.92% | 59.23% | +2.69 | 61.92% | 56.91% | 81.28% | 59.24% | 53.22% |
| 456 | control | 83.88% | 63.56% | 56.86% | +6.70 | 63.56% | 63.53% | 83.92% | 60.52% | 56.54% |
| 456 | **sal** | 84.20% | **65.80%** | 57.37% | **+8.43** | **65.80%** | 63.99% | 83.44% | **63.04%** | 57.36% |
| 456 | drophead | 81.84% | 63.36% | 58.22% | +5.14 | 63.40% | 63.25% | 82.40% | 60.76% | 58.35% |

Sliced model: **630.8M → 525.8M parameters, 16.63% smaller** — the same fraction
as on ViT-B/16, as expected, since it is set by the head quota and not by depth.

## Paired comparisons (mean over 3 seeds, 95% CI, sign count)

| comparison | result |
|---|---|
| **control: aligned vs random** | **+5.17pp**, CI [−0.10, **+10.44**], 3/3 |
| drophead: aligned vs random | +4.64pp, CI [+0.28, +9.01], 3/3 |
| sal: aligned vs random | +7.14pp, CI [+1.95, +12.33], 3/3 |
| **SAL vs control, identical (aligned) set** | **+3.04pp**, CI [+1.25, +4.83], 3/3 |
| **SAL vs DropHead, identical (aligned) set** | **+2.55pp**, CI [+2.24, +2.85], 3/3 |
| DropHead vs control, identical (aligned) set | +0.49pp, CI [−1.03, +2.01], 2/3 — **not established** |
| the same **with INT4** — SAL vs DropHead, aligned | **+2.61pp**, CI [+1.89, +3.33], 3/3 |
| the same — SAL vs control, aligned | **+3.13pp**, CI [+1.54, +4.73], 3/3 |
| sliced SAL-aligned vs sliced DropHead-random | +4.69pp, CI [−1.69, +11.06], 3/3 |
| clean accuracy: SAL vs control | +0.15pp, CI [−0.44, +0.73], 2/3 |
| clean accuracy: DropHead vs control | −1.08pp, CI [−3.15, +0.99], 0/3 |

## 1. All three arms find SAL's head set easier. That part is selection.

The decisive cell is the first row, and on I-JEPA it has the **opposite sign** to
ViT-B/16: **+5.17pp** against **−5.14pp**. A model that never saw masking scores
5pp *better* on the 256 heads SAL happened to draw than on 256 random heads.

What makes this hard to dismiss is that it shows up in **every arm**, including
DropHead, which resamples a fresh set every step and therefore has no "own" set
at all: control +5.17, drophead +4.64, sal +7.14, all 3/3. An effect common to
three differently-trained models is a property of **the set**, not of any
training. On this architecture, SAL's draw landed on heads that are genuinely
less load-bearing.

The honest caveat on strength: at 3 seeds the t-multiplier is 4.303, so the
control's interval is [−0.10, +10.44] and **just** spans zero. The sign is
consistent 3/3 and the point estimate is large, but this is **not established at
3 seeds** — it is a strong signal that needs 10 to be a claim. The random draws
themselves have sd 6.3–9.3pp across 10 draws (SE ≈ 2.3pp per seed), which is the
dominant noise term here, exactly as `PHASE2A.md` found.

Why the two architectures would differ is not measured. ViT-H/14 has 512 heads
against 144 and is far more over-parameterised for a 100-class probe, so a random
half is more likely to contain a removable half; depth (32 vs 12 layers) means the
uniform quota spreads 8 heads per layer rather than 6 of 12. Architecture,
objective (JEPA vs supervised) and task headroom are all confounded. **Nothing
here separates them.**

## 2. The adaptation reproduces, at almost exactly the ViT-B/16 magnitude.

Subtract the common component and SAL still gains **+3.04pp over the control on
the identical set** (3/3) and **+2.55pp over DropHead on the identical set**
(3/3, CI [+2.24, +2.85] — the tightest interval in the table, the three seeds
falling within 0.24pp of each other).

That same-set figure is the cleanest comparison in the design: no arm is handed a
set the others did not face. On ViT-B/16 it was **+2.02pp** [+1.54, +2.50], 10/10.
On I-JEPA it is **+2.55pp** [+2.24, +2.85], 3/3. **The two architectures agree on
the adaptation to within 0.5pp.** What exp15 adds is a second, additive
selection term on I-JEPA that ViT-B/16 does not have.

So the right statement is no longer "adaptation, not selection" as a general
claim, but: **adaptation on both architectures, plus a selection component on
I-JEPA.** The product consequence is mild and in SAL's favour — the deployable
number (SAL on its own set against the alternatives) is *larger* here, not
smaller. What weakens is the *mechanistic* story, not the benefit.

## 3. Under INT4, SAL's edge is established here — unlike on ViT-B/16.

`PHASE2F.md` recorded the honest limit that SAL's edge over DropHead vanished
once INT4 was stacked on top (+0.49pp, CI [−0.39, +1.37], 6/10). On I-JEPA it
survives: **+2.61pp**, CI [+1.89, +3.33], 3/3, and +3.13pp over the control.

INT4 costs the same ~2.9pp on both arms (−2.88pp on SAL-aligned, −2.97pp on
control-aligned, 0/3 both), so it is not differentially damaging — it simply does
not erase a 2.6pp gap the way it erased a 1.4pp one. This does **not** overturn
the ViT-B/16 non-result; the two measure different models and both stand.

## 4. DropHead is not the strong general method here.

On ViT-B/16 DropHead was the headline (+49.77pp over a standard fine-tune at
P50+I4) because the *baseline there collapsed* — 31.56% aligned against a 93.77%
clean. On I-JEPA nothing collapses: the control keeps 61–69% of an 83% clean
score under half its heads removed, a drop of ~18.9pp rather than ~62pp.

In that regime DropHead buys almost nothing: **+0.49pp over the control on the
identical set, 2/3, interval spanning zero**, while costing **−1.08pp of clean
accuracy** (0/3). SAL costs no clean accuracy (+0.15pp, CI [−0.44, +0.73]) and
gains 3.04pp. **On I-JEPA, SAL dominates DropHead on every cell measured.** The
v0.6.0 recommendation table — DropHead when the head set is out of your hands — is
built on the ViT-B/16 collapse regime and should not be read as architecture-free.

## 5. Slicing is exact on ViT-H/14 too.

`sliced_aligned` matches `aligned` to **≤0.16pp across all 9 arm×seed cells**
(seven of nine are identical to the printed precision), at 16.63% fewer
parameters. The masked result is a real smaller model on a 630M-parameter
encoder, not only on an 86M one.

## What is not established

- **The selection effect itself**, at 3 seeds: CI [−0.10, +10.44]. Sign 3/3,
  interval touching zero. Ten seeds would settle it; it costs ~32 GPU-h.
- **Why the two architectures differ.** Head count, depth, pretraining objective
  and task headroom are all confounded. No experiment here isolates any of them.
- **That I-JEPA's INT4 result transfers back to ViT-B/16.** It does not; the
  ViT-B/16 non-result at +0.49pp stands unchanged.
- **Anything at mask_ratio 0.3** on this architecture. Only 0.5 was run.
- **The `sliced_random` column as a comparison.** It averages 3 draws against the
  10 of `random`, with per-draw sd of 6–9pp, so differences between those two
  columns are within sampling noise and are not interpreted above.

## Caveat on an implementation detail

DropHead's arm took 1871.7 s against SAL's 990.2 s and the control's 997.8 s.
That is **not** a property of the method: in `exp15_ijepa_aligned.py` DropHead
runs through a hand-written loop that resamples each step, while SAL goes through
`SALTrainer` and the control through `run_jepa_sal.train_control`. The gap is the
loop, not the masking. It is recorded here only so the number is not mistaken for
a cost of DropHead.
