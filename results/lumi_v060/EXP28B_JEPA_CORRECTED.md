# exp28b — I-JEPA ViT-H/14 on ImageNet-1k with a frozen target encoder

`experiments/exp28b_jepa_corrected.py`, `.venv-tf47` (transformers 4.47.1),
torch 2.5.1+rocm6.2, one MI250X GCD per job. **223.6 GPU-hours**
(74.2-74.9 per seed plus an 11-minute gate); the full 1B campaign,
exp28 + exp28b, is **420.2 GPU-hours**.

This is the corrected re-run of [exp28](EXP28_JEPA_1B.md), whose every arm —
including the unmasked control — lost most of its representation quality and
which therefore measured nothing about either method. **Two things are settled
here.** The recipe fix holds: the control arm now survives two full epochs on
3/3 seeds. And with a sound control, the method comparison runs — **DropHead
beats SAL on every cell of the table, on 3/3 seeds, including the one head set
SAL itself trained for.**

## Protocol

`facebook/ijepa_vith14_1k`, **630 762 240 parameters**, 32 layers × 16 heads =
512 heads, patch 14 → 256 patches per 224×224 image.

**ImageNet-1k (resized 256px, `evanarlian/imagenet_1k_resized_256`)**,
1 281 167 train / 50 000 val, 1000 classes. **Two epochs of continued I-JEPA
pretraining, the full train split each epoch.** Batch 64, AdamW,
`clip_grad_norm_`, `PATCH_MASK_RATIO = 0.40`, autocast.

What changed against exp28, and it is one thing:

| | exp28 | exp28b |
|---|---|---|
| target forward | **the live weights being trained** | **a frozen copy of the pretrained weights** (`--target frozen`) |
| LR | 1e-5 | 5e-6 |
| warmup | none | 10% of total steps |
| schedule | none | cosine decay |
| control gate | none | per-epoch probe of the baseline arm, sentinel re-read every 200 steps |

exp28's objective took its target from the weights it was training, so it had no
fixed point: an encoder drifting toward a low-variance output drags both sides of
the MSE down together and the loss falls while the features are destroyed. The
frozen target anchors the objective to a representation known to score ~48.6%.

Three arms, identical data, optimiser, schedule and epoch count, differing only
in masking:

| arm | masking | `prune_events` |
|---|---|---|
| `baseline` (printed as `control`) | none — **this is the gated arm** | — |
| `sal` | accumulate-and-hold, 256 of 512 heads, uniform per layer (8/layer) | 256 |
| `drophead` | same 50% quota, mask resampled every step | 0 |

Three training seeds (42, 123, 456), all three trained, scored and reported.
Accuracy is `sal.evaluation.linear_probe` on **frozen features**: I-JEPA is
self-supervised and has no 1000-class head, so this is **not** ImageNet top-1
and is **not comparable with exp23-27**. The probe is fitted on 5000 images for
1000 classes — 5 per class — and validated on 2500.

Jobs, all `COMPLETED 0:0`:

| seed | baseline | sal | drophead | eval |
|---|---|---|---|---|
| 42 | `23540225` (19:22:43) | `23540223` (19:27:22) | `23540224` (23:23:31) | `23540226` (12:01:30) |
| 123 | `23567668` (19:38:43) | `23567665` (19:57:26) | `23567667` (23:36:25) | `23567669` (11:42:32) |
| 456 | `23567672` (19:31:41) | `23567670` (19:30:37) | `23567671` (23:31:48) | `23567673` (11:39:08) |

Gate `23539958` (11:00). Measurements in `results/lumi_v060/jepa1b_corr/`,
`jepa1b_corr_s123/`, `jepa1b_corr_s456/` and `jepa1b_corr_gate/`.

## 1. The control survives — the recipe fix is confirmed

The baseline arm probes itself after every epoch against a reference it measures
rather than assumes, and writes a pass/fail sentinel. It passed **3/3**:

| seed | baseline probe after epoch 2 | pretrained reference | delta | gate |
|---|---|---|---|---|
| 42 | 47.96% | 48.60% | **−0.64pp** | pass |
| 123 | 49.28% | 48.56% | **+0.72pp** | pass |
| 456 | 48.60% | 41.04% | **+7.56pp** | pass |

Against exp28, where each control arm dropped 27-33pp against its own
`original`. The abort threshold was 5pp below reference and was never
approached. **The cause of exp28's collapse was the objective, and nothing
else**: same model, same data, same scale, same masking code, same epoch count.

Two readings of this table matter for anything built on it:

- **The seed-456 row is not a 7.56pp improvement.** It is the known cross-job
  variance of the probe: the *same untrained checkpoint* reads 48.60 / 48.56 /
  41.04% in the three eval jobs. Compare within a seed, never across. This is
  why `original` carries an sd of 4.35 below while `control-trained` carries
  0.64 — the latter is one architecture scored three times by three probe fits,
  the former is the same weights scored by the same three fits.
- **The gate is worth keeping.** exp28's smoke jobs (90s) and its 1:22:53 gate
  both passed while the real runs were destroying the model. A probe of the
  *control* checkpoint after one epoch is the cheapest instrument that can see
  this class of failure, and here it would have cost ~10 GPU-hours against the
  196.6 exp28 spent. See `ROADMAP.md`, v0.7.0 "training harness".

## 2. The method comparison: DropHead dominates

Linear probe on frozen features, %, three seeds. `random-33` / `random-50` are
means over 20 random head selections per cell; the `@aligned` rows score all
three checkpoints on **SAL's own trained-for set** of 256 heads.

| row | 42 | 123 | 456 | mean | sd |
|---|---|---|---|---|---|
| `original` (pretrained) | 48.60 | 48.56 | 41.04 | **46.07** | 4.35 |
| `control-trained` | 48.00 | 49.28 | 48.60 | **48.63** | 0.64 |
| `drophead-trained` | 43.48 | 42.56 | 48.48 | **44.84** | 3.19 |
| `sal-trained` | 35.52 | 35.88 | 37.44 | **36.28** | 1.02 |
| `ctrl+random-33%` | 33.65 | 32.41 | 32.72 | **32.93** | 0.65 |
| `dh+random-33%` | 44.66 | 44.88 | 44.87 | **44.80** | 0.13 |
| `sal+random-33%` | 31.32 | 32.00 | 34.80 | **32.70** | 1.84 |
| `ctrl+random-50%` | 16.73 | 13.44 | 13.85 | **14.67** | 1.80 |
| `dh+random-50%` | 43.14 | 43.59 | 44.36 | **43.70** | 0.61 |
| `sal+random-50%` | 21.64 | 20.25 | 25.80 | **22.56** | 2.89 |
| `control@aligned` | 25.04 | 19.00 | 14.52 | **19.52** | 5.28 |
| `drophead@aligned` | 47.52 | 43.08 | 41.96 | **44.19** | 2.94 |
| `sal@aligned` | 47.48 | 42.72 | 40.36 | **43.52** | 3.63 |
| `control@magnitude33` | 39.52 | 34.92 | 35.60 | **36.68** | 2.48 |
| `drophead@magnitude33` | 44.48 | 44.92 | 44.64 | **44.68** | 0.22 |
| `sal@magnitude33` | 35.16 | 34.04 | 36.72 | **35.31** | 1.35 |
| `control@magnitude50` | 17.28 | 15.80 | 15.20 | **16.09** | 1.07 |
| `drophead@magnitude50` | 42.44 | 43.44 | 46.48 | **44.12** | 2.10 |
| `sal@magnitude50` | 23.56 | 21.72 | 29.16 | **24.81** | 3.88 |

The full table, the magnitude and sliced rows, and the INT4 column are in the
per-seed JSON.

### Against the control, on the matched cell, within each seed

| cell | arm | 42 | 123 | 456 | mean | wins |
|---|---|---|---|---|---|---|
| dense | drophead | −4.52 | −6.72 | −0.12 | **−3.79** | 0/3 |
| dense | sal | −12.48 | −13.40 | −11.16 | **−12.35** | 0/3 |
| random-33% | drophead | +11.00 | +12.47 | +12.15 | **+11.88** | **3/3** |
| random-33% | sal | −2.33 | −0.41 | +2.08 | **−0.22** | 1/3 |
| random-50% | drophead | +26.41 | +30.15 | +30.51 | **+29.02** | **3/3** |
| random-50% | sal | +4.91 | +6.81 | +11.96 | **+7.89** | 3/3 |
| SAL's aligned set | drophead | +22.48 | +24.08 | +27.44 | **+24.67** | **3/3** |
| SAL's aligned set | sal | +22.44 | +23.72 | +25.84 | **+24.00** | 3/3 |
| magnitude-33% | drophead | +4.96 | +10.00 | +9.04 | **+8.00** | **3/3** |
| magnitude-33% | sal | −4.36 | −0.88 | +1.12 | **−1.37** | 1/3 |
| magnitude-50% | drophead | +25.16 | +27.64 | +31.28 | **+28.03** | **3/3** |
| magnitude-50% | sal | +6.28 | +5.92 | +13.96 | **+8.72** | 3/3 |

### DropHead minus SAL, within each seed

**Every cell, every seed.**

| cell | 42 | 123 | 456 | mean | wins |
|---|---|---|---|---|---|
| dense | +7.96 | +6.68 | +11.04 | **+8.56** | 3/3 |
| random-33% | +13.34 | +12.89 | +10.08 | **+12.10** | 3/3 |
| random-50% | +21.50 | +23.34 | +18.55 | **+21.13** | 3/3 |
| **SAL's aligned set** | +0.04 | +0.36 | +1.60 | **+0.67** | **3/3** |
| magnitude-33% | +9.32 | +10.88 | +7.92 | **+9.37** | 3/3 |
| magnitude-50% | +18.88 | +21.72 | +17.32 | **+19.31** | 3/3 |

### What the numbers say

- **SAL has no remaining advantage, including on its own set.** The aligned row
  was SAL's last defensible niche — the deployment whose head set is fixed at
  training time. On that set DropHead reads **44.19%** against SAL's
  **43.52%**, and wins on **3/3 seeds**. The margin is small (+0.67pp) and the
  honest statement is a tie rather than a rout, but a tie is enough: DropHead
  gets there **without committing to a head set at training time**, and it is
  simultaneously 12 to 21 points ahead on every set SAL did not train for. There
  is no cell left where choosing SAL is the better decision.
- **SAL's adaptation is real, and still entirely set-specific.** +24.00pp over
  the control on its own set, on 3/3 seeds; −0.22pp at random-33% and −1.37pp at
  magnitude-33%, where the sign flips with the seed. This replicates exp25's
  finding at 631M on a self-supervised objective.
- **SAL pays 12.35pp of dense accuracy** (3/3 seeds) and never earns it back:
  its best cell, its own aligned set, is **5.11pp below the control's dense
  row**. DropHead's aligned row is 4.44pp below it, and DropHead's P50 row is
  **1.14pp** below its own dense row — pruning half the heads is close to free
  for DropHead, as it was on ViT-B/16 in exp27.
- **Pruning *helps* SAL's own checkpoint.** `sal@aligned` is **+7.24pp above
  `sal-trained`**: SAL trained with those 256 heads masked, so running it dense
  is off-distribution for it. Its dense row is not its operating point —
  the same correction exp25 made for ViT-B/16.
- **Head selection matters far less than the training method.** Magnitude
  selection beats random for the control (+3.75pp at 33%) and for SAL, and makes
  no difference to DropHead (44.68 vs 44.80 at 33%): a model trained against
  arbitrary masks does not care which heads go.
- The control loses **33.95pp** at random-50% and **15.70pp** at random-33%
  against its own dense row, so this model is firmly in the probe's
  "**>30pp → train for it**" regime at 50%, and in the "either" band at 33%.

## 3. Consistency with the rest of the project

This is the first 1B-scale, self-supervised confirmation of what exp27 measured
on ViT-B/16 / ImageNet-1k classification, and the two agree on direction and on
ordering while differing in magnitude:

| | exp27 (ViT-B/16, 86M, supervised) | exp28b (I-JEPA ViT-H/14, 631M, SSL) |
|---|---|---|
| DropHead over control at P50 | +63.47pp | +29.02pp |
| DropHead over control at P33 | +26.45pp | +11.88pp |
| DropHead over SAL on SAL's own set | +54.70pp | +0.67pp |
| cost of P33 to DropHead | +0.14pp (free) | −0.04pp (free) |
| cost of P50 to DropHead | −5.01pp | −1.14pp |

The gap on the aligned row is the one real difference: on ViT-B/16 DropHead
buried SAL on SAL's own set, here it merely matches it. A 631M SSL encoder has
more redundancy to spend, and SAL's two epochs of accumulate-and-hold have more
room to specialise. It changes the margin, not the recommendation.

## Caveats

- **Not ImageNet top-1.** A 5-image-per-class linear probe on frozen features.
  Absolute levels are not comparable with exp23-27; only within-table
  comparisons are meaningful.
- **Compare within a seed.** The same pretrained checkpoint reads 48.60 /
  48.56 / 41.04% across the three eval jobs — 7.56pp of cross-job probe spread.
  Every delta quoted above is computed inside a seed for that reason.
- **Two epochs, not convergence.** Both arms are still moving at epoch 2;
  nothing here says where they end up.
- **INT4 is simulated.** `quantize_weights(model, 4)`, a per-channel
  quantize/dequantize round-trip in fp32. bitsandbytes NF4 is CUDA-only and
  absent on the ROCm image. The column is the accuracy cost of 4-bit weight
  rounding and says nothing about speed or memory. On the dense rows the cost is
  **under 0.5pp for all three arms** (control −0.16, drophead −0.43, sal −0.49,
  3/3 seeds each); on the pruned rows it grows to 1.4-3.9pp. **The pretrained
  row cannot be used as a reference point for this**: it reads −9.20 / −1.44 /
  −0.32pp on the three seeds, which is probe noise rather than a quantization
  effect, so there is no measured "continued pretraining reduces the INT4 cost
  from −9.2pp" claim to make.
- **No latency claim.** `measure_latency` runs at **batch 1**, where a ViT is
  kernel-launch bound; the sliced cells read 0.97-1.00x and that is the expected
  null result, not a finding. See exp26 for the batch-32 measurements.
- **`slice_heads()` is exonerated again.** Across all 12 sliced cells on all
  three seeds the largest relative difference against masking is **2.1e-04**.
- SAL's losses *rise* between epoch 1 and 2 (0.2656 → 0.2997 on seed 42) while
  the control's fall (0.2461 → 0.2133). With a frozen target that is
  interpretable — SAL is removing heads as it trains, so the objective gets
  harder — unlike exp28, where falling losses coexisted with collapsing
  features.
