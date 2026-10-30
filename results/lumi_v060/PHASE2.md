# v0.6.0 Phase 2 — exp04: SAL+QAT on ViT-B/16, five seeds on MI250X

ViT-B/16 (`google/vit-base-patch16-224`, 100-way head) on ImageNet-100,
20,000 train / 5,000 val, 5 epochs, batch 64, `mask_ratio=0.3`, QAT at 4-bit
via the native `torch.ao` backend. Each compression cell is a mean over
3 random head selections, identical across arms. One GCD per seed, run in
parallel; 23–30 min each, **~2.2 GPU-hours** for all five.

Seeds 42/123/456 reproduce the v0.5.0 NVIDIA runs. Seeds 137/2024 are new.

## Two conclusions, and they point in different directions

### 1. ROCm reproduces NVIDIA

On the three shared seeds the key cell lands within ~1pp of the published
figure, which is inside this experiment's own noise:

| seed | ROCm (MI250X) | NVIDIA (v0.5.0) | difference |
|---|---|---|---|
| 42 | +15.52pp | +15.51pp | +0.01 |
| 123 | +26.17pp | +25.28pp | +0.89 |
| 456 | +27.21pp | +25.13pp | +2.08 |

Mean difference **+0.99pp**. The platform is not a confound.

### 2. The +22pp headline does not survive two more seeds

| seed | baseline P50+I4 | SAL+QAT P50+I4 | delta |
|---|---|---|---|
| 42 | 36.5% | 52.0% | **+15.52pp** |
| 123 | 32.5% | 58.7% | **+26.17pp** |
| 456 | 23.6% | 50.8% | **+27.21pp** |
| 137 | 26.2% | 37.4% | **+11.21pp** |
| 2024 | 24.5% | 25.0% | **+0.57pp** |

**Mean +16.13pp, sd 11.07, positive on 5/5 seeds.**

The published claim is "+22.0pp mean over 3 seeds". Those three seeds were the
three most favourable of the five. With 137 and 2024 added the mean falls to
+16.1pp and the standard deviation (11.07) is two thirds of the mean. The sign
is robust — SAL+QAT wins every seed — but **the magnitude is not a stable
quantity and should not be quoted as a single number.**

Seed 2024 is worth looking at directly, because it is not merely a small
effect, it is a different shape. There, SAL *hurts* under head pruning alone:

```
seed 2024        FP16  Prune33%  Prune50%    INT8    INT4   P50+I4
Baseline        93.5%     64.9%     27.9%   93.5%   92.9%    24.5%
SAL only        93.1%     60.0%     29.2%   93.2%   92.7%    26.7%
SAL + QAT       93.2%     55.1%     25.8%   93.2%   92.7%    25.0%
```

At Prune33% SAL is 4.9pp *below* baseline and SAL+QAT is 9.8pp below. This is
the first observation in the project of SAL being actively harmful on a
configuration it is supposed to help.

### Where the variance comes from

It is not only the training seed. Absolute P50+I4 accuracy for the **SAL-only**
arm spans 26.7% to 56.4% across seeds — a 30pp range on one fixed
configuration. The P50 regime is intrinsically high-variance: half the heads of
a 12-layer model are removed, and each cell averages only **3** random head
selections (`--prune-seeds 3`). Much of the spread in the delta is the baseline
and SAL arms each being unstable, not a clean signal moving.

That suggests a cheap fix before anything else: raising `--prune-seeds` costs
only evaluation passes, no extra training, and it is the dominant noise term.

## The QAT increment holds up better

Adding QAT on top of SAL, at P50+I4: +3.21, +2.25, +4.17, +5.31, **-1.68** pp —
mean **+2.65pp**, against the published +3.1pp from 3 seeds. Smaller and with
one negative seed, but far more stable in relative terms than the headline
number.

## Unchanged, and worth noting

The FP16 / INT8 / INT4 columns are flat across all arms and all seeds
(93.0–93.9% / 93.1–93.9% / 92.4–93.4%). SAL costs ~0.6pp of clean accuracy and
INT8 remains free. Nothing here contradicts v0.5.0: **quantization alone is not
where SAL pays, and head pruning is.**

## Consequences

- The "+22pp" figure in `CLAUDE.md` (and the same claim in sal-torch-pro's
  `CLAUDE.md`) is now a 3-seed subsample of a 5-seed result. It should be
  restated as **+16pp mean, sd 11, 5/5 seeds positive**, or better, reported as
  a range with the per-seed table — which is what this file is for.
- Any further SAL+QAT experiment should raise `--prune-seeds` well above 3
  before adding training seeds; the head-selection average is the cheapest
  variance to buy down.
- Seed 2024 deserves a direct look rather than being averaged away: if there is
  a model/seed state where progressive head masking does net harm, that is a
  limit of the method and belongs in the documentation.
