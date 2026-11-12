# exp33 — SparseGPT / Wanda against DropHead and magnitude

**STUB — no accuracy results yet.** The ROCm check and the equivalence test are
done and recorded below; the tables arrive with the runs.

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

| row | training epochs | structure | speedup @ batch >= 32 | accuracy (P50) |
|---|---|---|---|---|
| dense control, nothing pruned | 5 | — | 1.00x | **80.556%** (exp31) |
| pretrained checkpoint | 0 | — | 1.00x | 78.44% |
| DropHead, **no recovery** | 5 (DropHead) | structured | **1.10x** | **72.54%** (exp27) |
| DropHead → magnitude + recovery | 5 + 5 | structured | **1.10x** | **77.06%** (exp24 arm 4) |
| magnitude + recovery, 5 ep | 5 | structured | **1.10x** | 77.265% (exp29) |
| magnitude + recovery, **2 ep** | **2** | structured | **1.10x** | **77.570%** (exp29) |
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

## Jobs

| seed | job | state |
|---|---|---|
| | | _pending submission_ |
