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

## Structured references

Measured on the identical model, dataset and eval path — printed beside this
run's numbers, never reused as data:

| arm | accuracy | source |
|---|---|---|
| DropHead at P33 | 77.69% | `EXP27_DROPHEAD.md` |
| DropHead at P50 | 72.54% | `EXP27_DROPHEAD.md` |
| magnitude + 5 ep at P33 | 78.845% | exp29 |
| magnitude + 5 ep at P50 | 77.265% | exp29 |
| magnitude + 2 ep at P50 | 77.570% | exp29 |
| dense control, 5 ep, nothing pruned | 80.556% | exp31 |

## Jobs

| seed | job | state |
|---|---|---|
| | | _pending submission_ |
