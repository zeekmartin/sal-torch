# exp33 — SparseGPT / Wanda against DropHead and magnitude

**PARTIAL — the 2:4 arm is complete at 3 seeds (below); the unstructured grid
is running.** The ROCm check and the equivalence test are done and recorded
below.

**ViT-B/16 (`google/vit-base-patch16-224`), ImageNet-1k
(`evanarlian/imagenet_1k_resized_256`), 3 seeds, full 50 000-image validation
split, autocast bfloat16.** Script `experiments/exp33_sparse_vs_structured.py`,
pruning cores in `experiments/exp33_sparse_core.py`.

## Why

Every baseline this project has measured head methods against is **structured**
— magnitude head pruning, with and without recovery. The strongest published
one-shot pruners are **unstructured**: SparseGPT (ICML 2023) and Wanda (ICLR
2024). At a matched count of removed parameters, do they beat training-time
head masking? If they do, the routing table needs another row.

## Provenance of the pruning code

Both algorithms are **vendored**, not imported, so nothing third-party runs on
the allocation and the code we ship is reviewable here:

- **SparseGPT** — https://github.com/IST-DASLab/sparsegpt, **Apache-2.0**
- **Wanda** — https://github.com/locuslab/wanda, **MIT**, © 2023 CMU Locus Lab

`tests/test_exp33_sparse_equivalence.py` compares both against the upstream
code at layer level — same layer, same calibration batch. **Result: identical
masks, weights agreeing to 1e-5 (SparseGPT) and 1e-6 (Wanda), and matching
Hessian / activation-norm accumulation.** Six invariant tests run in CI; the
two comparison tests need `SAL_SPARSE_REFS` pointing at clones of the two
repos, since their code is not vendored.

The papers' numbers are **not** the correctness check: they are LLM
perplexities on WikiText, from models and a calibration pipeline we are not
running. A matching perplexity would not show the ViT pass is right, and a
differing one would not show it is wrong. The layer-level equivalence test is
the check.

## ROCm compatibility: PASS

Neither reference needs a custom CUDA kernel — the whole pruning path is torch
ops. The only CUDA-specific line in either core is an unconditional
`torch.cuda.synchronize()` at the end of SparseGPT's `fasterprune`, a barrier
rather than arithmetic, which works on ROCm because torch maps `torch.cuda`
onto HIP. The one `bitsandbytes` import in the Wanda repo is in
`dense_ft/trainer.py`, a vendored copy of HuggingFace's `Trainer` for an
optional fine-tuning path, not in the pruning core.

Measured on an MI250X GCD (job `23669807`, torch 2.5.1+rocm6.2, raw in
`results/exp33_rocm_check.json`):

| check | result |
|---|---|
| `torch.linalg.cholesky` + `cholesky_inverse`, 2048×2048 | **ok**, 1.99 s |
| Wanda over all 48 ViT-B/16 attention projections, 128 calib images | **ok**, 39.9 s |
| SparseGPT, same | **ok**, 7.4 s |
| zero fraction over target layers | 0.5000 for both, every weight finite |

The ROCm image needs no workaround.

## Comparison convention

Every pruned arm is scored against **the dense control trained for the same
number of epochs**, never against the pretrained checkpoint, and the two
effects are reported separately:

- **fine-tuning gain** = dense control − pretrained — what the epochs bought.
- **pruning cost** = arm − matched dense control — what the compression cost.

The checkpoint is not a control: an arm that was fine-tuned has had epochs the
checkpoint never had, so scoring against it reports the sum of the two and can
make a model that pruning damaged look improved. Where an arm has one seed it
is scored against **that seed's own** control, not the three-seed mean (exp28
measured a 7.56pp spread from evaluation shuffling alone).

Full ladder and sources in `CONTROL_CONVENTION.md`. For ViT-B/16 on
ImageNet-1k:

| dense control | top-1 | INT4 | seeds | fine-tuning gain |
|---|---|---|---|---|
| 0 ep — pretrained | 78.44% | 77.23% | deterministic | — |
| **2 ep** | **81.554%** | **80.273%** | 3, sd 0.11 | **+3.11pp** |
| 5 ep | 80.44% | 79.01% | 3, sd 0.05 | +2.00pp |

**This experiment mixes epoch counts on purpose**, so the convention is load
bearing here: the one-shot arms pay 0 epochs and go against the **pretrained**
row, the recovery arms pay 2 and go against the **2-epoch** control, and the
DropHead rows quoted from exp27 paid 5 and go against the **5-epoch** one. Each
row in the final table names its own epoch count for exactly this reason.

## Design

| arm | what is pruned | matched how |
|---|---|---|
| `attn` | the q/k/v/o projections only | removing a fraction *f* of heads uniformly per layer removes exactly *f* of the q/k/v/o weight parameters, so the matched sparsity **is** *f*. Confirmed on the GPU: 50% → 14 155 776 of 28 311 552 target params. |
| `whole` | every encoder linear layer, attention and MLP; classifier excluded | the same **absolute** parameter count, which at the 50%-of-heads equivalent is 16.35% of the model and so a much lower per-layer sparsity |

Each cell is scored **one-shot** and **after 2 epochs of recovery** (exp29).
The three seeds vary **which 128 training images calibrate** — the only
stochastic input the one-shot arms have — and, for the recovery arms, the data
order.

## Two traps this experiment has to avoid

1. **Recovery destroys unstructured sparsity unless the mask is enforced.** An
   ordinary fine-tune fills the zeros straight back in, and the arm silently
   stops being sparse while still carrying the label "50% pruned". Every
   recovery here re-applies the prune-time mask through a gradient hook plus a
   final re-apply, **re-measures the zero fraction afterwards, and records
   `mask_preserved` and `sparsity_drift` in the JSON.** A drifting mask
   invalidates the cell, and the run says so instead of reporting a dense model
   as sparse.
2. **There is no speedup to report for the unstructured arms.** The parameters
   are masked, not removed, so a dense kernel does exactly as much arithmetic
   as before. Latency is reported for the structured methods only, at batch 32
   (exp26: at batch 1 every arm on this model is 1.00x). An empty speedup
   column would read as "not measured" rather than "does not exist", so the
   table has to state it.

## The final table: every row states its own terms

The comparison is only meaningful if each row says what it cost to produce and
what it can claim. **Every row carries four things**: training cost in epochs,
structure class, speedup at batch >= 32, and accuracy. Rows are never compared
across different training costs without the difference being visible.

### Why this matters: the 72.54% is not what it looks like

`EXP27_DROPHEAD.md`'s **72.54%** is DropHead **without any post-pruning
recovery** — the model is trained with per-step resampled masking for 5
epochs, then the P50 magnitude head set is removed and it is scored as-is. Put
beside a one-shot sparse arm it looks like a fair fight (neither had a recovery
pass) and beside `magnitude + 2 ep` it does not (one had recovery, the other
did not, and DropHead paid 5 epochs of training the other never paid).

So the table carries **both DropHead rows**. The recovery counterpart is
already measured: exp24's arm 4, `DropHead -> magnitude + recovery`, reads
**77.06% / 75.13% INT4** at P50 — which is **−0.21pp against plain
magnitude + recovery** from the pretrained checkpoint, i.e. DropHead's weights
buy nothing as a pre-conditioner, the finding exp24 was run to establish.

### Row template

Every row also names the control it is scored against, which is the row above
it with the same epoch count.

| row | training epochs | structure | speedup @ batch >= 32 | accuracy (P50) |
|---|---|---|---|---|
| **control: pretrained** | **0** | — | 1.00x | **78.44%** |
| **control: dense + 2 ep** | **2** | — | 1.00x | **81.554%** (3 seeds) |
| **control: dense + 5 ep** | **5** | — | 1.00x | **80.44%** (exp25) / 80.556% (exp31) |
| _fine-tuning gain_ | 2 ep | — | — | _+3.11pp_ (5 ep: +2.00pp) |
| DropHead, **no recovery** | 5 (DropHead) | structured | **1.10x** | **72.54%** (exp27) |
| DropHead → magnitude + recovery | 5 + 5 | structured | **1.10x** | **77.06%** (exp24 arm 4) |
| magnitude + recovery, 5 ep | 5 | structured | **1.10x** | 77.265% (exp29) → **−3.17pp** |
| magnitude + recovery, **2 ep** | **2** | structured | **1.10x** | **77.642%** (exp32, 3 seeds) → **−3.91pp** |
| SparseGPT attn, one-shot | **0** | unstructured | **none by construction** | _this run_ |
| SparseGPT attn, + 2 ep | **2** | unstructured | **none by construction** | _this run_ |
| Wanda attn, one-shot | **0** | unstructured | **none by construction** | _this run_ |
| Wanda attn, + 2 ep | **2** | unstructured | **none by construction** | _this run_ |
| SparseGPT attn **2:4**, one-shot | **0** | **semi-structured** | **unmeasurable here** | _this run_ |
| SparseGPT attn **2:4**, + 2 ep | **2** | **semi-structured** | **unmeasurable here** | _this run_ |
| Wanda attn **2:4**, one-shot | **0** | **semi-structured** | **unmeasurable here** | _this run_ |
| Wanda attn **2:4**, + 2 ep | **2** | **semi-structured** | **unmeasurable here** | _this run_ |

The whole-model unstructured arms join the same table at their own matched
parameter count, and the P33 arms repeat it at that ratio.

**The trap this closes, concretely.** Wanda attention-P33 plus 2 epochs reads
**81.60%** on seed 42. Against the 5-epoch control (80.556%) that is +1.04pp
and reads as "unstructured sparsity beats dense". Against its own matched
2-epoch control (seed 42: 81.676%) it is **−0.08pp** — level with dense, which
is a strong result for a method that removes 9.3M parameters, and a different
claim entirely. The one-shot cell of the same arm reads 78.43% against the
pretrained 78.44%, i.e. **free at zero epochs**, and that is the row the
pretrained control belongs to.

### The three speedup entries are three different claims

- **1.10x** — measured, at batch 32, for removing 50% of heads (exp26). At
  batch 1 every arm on this model is 1.00x, so the batch size is part of the
  claim.
- **none by construction** — the unstructured arms mask weights instead of
  removing them, so a dense kernel does identical arithmetic. This is not an
  unmeasured quantity; it is zero, and saying "n/a" would understate it.
- **unmeasurable here** — 2:4 *can* be accelerated, by NVIDIA sparse tensor
  cores on Ampere or newer with a kernel that exploits the pattern. LUMI is
  MI250X and ROCm has no equivalent path, so **this campaign cannot put a
  number on it**. Reporting it as "none" would be wrong, and reporting a number
  from someone else's hardware would be worse.

## The 2:4 arm

SparseGPT and Wanda, **attention projections only**, one-shot and + 2 epochs,
3 seeds — 12 runs, **~13 GPU-h**. 2:4 is inherently 50% sparse, so it has no
ratio of its own and lands against the P50 rows by construction.

Both cores take the references' own n:m path: the mask is built per group of 4
consecutive columns within each output row, by `topk(metric, 2, largest=False)`.
For SparseGPT that happens **inside** the column loop, against the
already-error-corrected weights rather than a metric computed up front, which
is the method rather than an implementation detail and is reproduced as such.
`tests/test_exp33_sparse_equivalence.py` checks both against upstream at 2:4 as
well as at unstructured sparsity, plus an invariant that exactly two of every
four weights are zero. **13 tests pass.**

### Results: 2:4 is complete, 3 seeds

12 runs, 11.71 GPU-hours, jobs `23672190/91/92`, all `COMPLETED 0:0`. Data in
`exp33_nm_s{42,123,456}/exp33_sparse.json`. 14 155 776 of 28 311 552 attention
parameters zeroed, **zero fraction exactly 0.5000** and **sparsity drift exactly
0.0 with `mask_preserved` true on 6/6 recovery runs** — the mask survived
fine-tuning, so these cells are what they are labelled.

| arm | epochs | seed 42 | seed 123 | seed 456 | mean | vs matched control |
|---|---|---|---|---|---|---|
| Wanda 2:4, one-shot | **0** | 72.89% | 72.94% | 72.79% | **72.871%** | **−5.57pp** vs pretrained |
| SparseGPT 2:4, one-shot | **0** | 75.10% | 75.39% | 75.18% | **75.223%** | **−3.22pp** vs pretrained |
| Wanda 2:4, + 2 ep | **2** | 80.99% | 81.03% | 80.98% | **80.999%** | **−0.55pp** vs 2-ep control |
| SparseGPT 2:4, + 2 ep | **2** | 81.08% | 81.08% | 81.14% | **81.097%** | **−0.46pp** vs 2-ep control |

INT4 means: 71.188 / 73.919 one-shot, 79.729 / 79.779 recovered (−0.54 and
−0.49pp against the 2-epoch control's 80.273%). Seed spreads are 0.15pp and
below on every cell — the calibration sample is not a meaningful noise source
here, unlike the training seed in the SAL campaigns.

**SparseGPT's one-shot lead over Wanda is real and then it evaporates.**
+2.35pp at zero epochs — the error compensation earning its keep under the
tighter 2:4 constraint, where the two are within 0.1pp at unstructured
sparsity — and **+0.10pp after two epochs of recovery**, which is inside the
seed spread. If recovery is affordable the choice between them does not matter;
if it is not, SparseGPT is worth 2.35pp.

### What 2:4 costs against structured head pruning

Both remove the same share of attention parameters (50%, 16.35% of the model),
so this is the comparison the experiment exists for:

| method at ~16.4% of params | epochs | structure | speedup @ batch ≥ 32 | vs 2-ep control |
|---|---|---|---|---|
| magnitude heads P50 | 2 | structured | **1.10x measured** | **−3.91pp** (exp32) |
| SparseGPT 2:4 | 2 | semi-structured | unmeasurable here | **−0.46pp** |
| Wanda 2:4 | 2 | semi-structured | unmeasurable here | **−0.55pp** |

**2:4 costs roughly an eighth of what head pruning costs**, and the thing it
gives up is the speedup this hardware can actually deliver. That is the
trade-off stated plainly, and it is why the speedup column may not be left
empty: a reader comparing only the accuracy column would conclude head pruning
is simply worse, when what it is, is the only one of the three that makes the
model faster *here*.

One caveat that belongs with every unstructured and 2:4 row: those models are
**the same size in memory**. The weights are zeroed, not removed, so there is no
memory saving and no latency change without a kernel that exploits the pattern.
The structured arm is genuinely 16.5% smaller.

## Jobs

| arm | seed | job | state |
|---|---|---|---|
| ROCm check | — | `23669807` | COMPLETED 0:02:49 |
| smoke | — | `23670445` | COMPLETED 0:07:57 |
| order check | — | `23670472` | COMPLETED 0:03:46 |
| 2:4 smoke | — | `23672122` | COMPLETED 0:01:46 |
| unstructured grid, 8 configs | 42 | `23671606` | RUNNING |
| unstructured grid | 123 | `23671607` | RUNNING |
| unstructured grid | 456 | `23671608` | RUNNING |
| **2:4, attention-only** | 42 | `23672190` | **COMPLETED** 3:55:05 |
| **2:4** | 123 | `23672191` | **COMPLETED** 3:55:24 |
| **2:4** | 456 | `23672192` | **COMPLETED** 3:52:03 |
