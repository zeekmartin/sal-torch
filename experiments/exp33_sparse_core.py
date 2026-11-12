#!/usr/bin/env python3
"""Vendored SparseGPT and Wanda pruning cores.

Both algorithms are reimplemented here, following the reference
implementations closely enough that a layer-level equivalence test passes
(``tests/test_exp33_sparse_equivalence.py``). They are vendored rather than
imported so that nothing third-party runs on the allocation and so the code we
actually ship is reviewable in this repo.

Attribution
-----------
**SparseGPT** — Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be
Accurately Pruned in One-Shot" (ICML 2023).
Reference: https://github.com/IST-DASLab/sparsegpt, **Apache License 2.0**.
``SparseGPTPruner`` follows ``sparsegpt.py::SparseGPT`` — the same Hessian
accumulation and rescaling, ``percdamp`` 0.01 damping on the diagonal, the
Cholesky / ``cholesky_inverse`` / upper-Cholesky route to ``Hinv``, a block
size of 128, per-block mask selection by ``w**2 / diag(Hinv)**2`` against a
per-block threshold, and the same two-level error propagation (within the
block column by column, then once onto the columns after it).

**Wanda** — Sun, Liu, Bair & Kolter, "A Simple and Effective Pruning Approach
for Large Language Models" (ICLR 2024).
Reference: https://github.com/locuslab/wanda, **MIT License**,
Copyright (c) 2023 CMU Locus Lab.
``WandaPruner`` follows ``image_classifiers/layerwrapper.py::WrappedLayer`` and
``image_classifiers/prune_utils.py::compute_mask`` — the metric is
``|W| * sqrt(scaler_row)`` with ``scaler_row`` the running sum of per-batch mean
squared column norms, and the comparison group is **per output row**, which is
the paper's default and what ``prune_granularity="row"`` selects.

Neither reference needs a custom CUDA kernel: the whole pruning path is torch
ops. The ROCm-sensitive ones are ``torch.linalg.cholesky``,
``torch.cholesky_inverse``, ``torch.sort`` and ``torch.topk``.

What this module deliberately does not do
-----------------------------------------
Both methods produce **unstructured** sparsity: zeros scattered inside the
weight matrices. The parameters are *masked, not removed*, so a dense kernel
does exactly as much arithmetic as before. **There is no speedup to report for
these arms** — compare them on accuracy at a matched count of zeroed
parameters, never on latency. ``slice_heads()`` and FFN neuron slicing are the
structured methods where size and latency actually move.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["SparseGPTPruner", "WandaPruner", "prune_model",
           "linear_layers", "attention_linear_layers", "sparsity_report"]

BLOCKSIZE = 128
PERCDAMP = 0.01


# ----------------------------------------------------------------- SparseGPT
class SparseGPTPruner:
    """One layer's SparseGPT state: accumulate a Hessian, then prune.

    Mirrors ``SparseGPT`` in the reference. ``add_batch`` keeps the reference's
    rescaling, which holds ``H`` at ``2/n * sum_i x_i x_i^T`` while ``n`` grows,
    so the damping term stays on the same scale whatever the calibration size.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        w = layer.weight.data
        self.rows, self.columns = w.shape[0], w.shape[1]
        self.H = torch.zeros((self.columns, self.columns),
                             device=w.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def prune(self, sparsity: float, blocksize: int = BLOCKSIZE,
              percdamp: float = PERCDAMP) -> int:
        """Prune to ``sparsity`` and write the weights back. Returns #zeroed."""
        W = self.layer.weight.data.clone().float()
        H = self.H

        # A column the calibration set never excited carries no information;
        # the reference zeroes it outright and keeps the Hessian invertible.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=W.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        zeroed = 0
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            zeroed += int(mask1.sum().item())

        self.layer.weight.data = W.reshape(
            self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.H = None
        return zeroed


# --------------------------------------------------------------------- Wanda
class WandaPruner:
    """One layer's Wanda state: accumulate column norms, then mask.

    Mirrors ``WrappedLayer`` plus ``compute_mask(..., "row", sparsity)``. Wanda
    never updates the surviving weights — it only zeroes — which is the whole
    difference in cost against SparseGPT.
    """

    def __init__(self, layer: nn.Linear, p_norm: int = 2):
        self.layer = layer
        w = layer.weight.data
        self.rows, self.columns = w.shape[0], w.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=w.device,
                                      dtype=torch.float32)
        self.nsamples = 0
        self.p_norm = p_norm

    def add_batch(self, inp: torch.Tensor):
        assert inp.shape[-1] == self.columns, (inp.shape, self.columns)
        inp = inp.reshape((-1, self.columns)).float()
        tmp = inp.shape[0]
        inp = inp.t()
        if self.p_norm == 2:
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / tmp
        else:
            self.scaler_row += torch.norm(inp, p=1, dim=1) / tmp
        self.nsamples += tmp
        if torch.isinf(self.scaler_row).any():
            raise ValueError("Wanda: non-finite activation norm — "
                             "calibration batch has an inf in it")

    def prune(self, sparsity: float) -> int:
        W = self.layer.weight.data
        metric = torch.abs(W).float() * torch.sqrt(
            self.scaler_row.reshape((1, -1)))
        # Per-output-row comparison group: each row loses the same fraction.
        mask = torch.zeros_like(metric, dtype=torch.bool)
        k = int(metric.shape[1] * sparsity)
        if k > 0:
            idx = torch.sort(metric, dim=-1, stable=True)[1][:, :k]
            mask.scatter_(1, idx, True)
        W[mask] = 0
        return int(mask.sum().item())


# ----------------------------------------------------------------- plumbing
def linear_layers(module: nn.Module, prefix: str = "") -> dict:
    """Every ``nn.Linear`` under ``module``, by dotted name."""
    out = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            out[f"{prefix}{name}" if prefix else name] = child
    return out


def attention_linear_layers(model: nn.Module) -> dict:
    """The Q/K/V/O projections only, via the package's own architecture map.

    Uses ``sal.arch_support`` so "attention projections" means the same thing
    here as it does to ``HeadMasker`` and the Fragility Index, rather than a
    second definition by name matching.
    """
    from sal.arch_support import get_attention_modules, get_output_projection
    from sal.arch_support import get_qkv_projections

    out = {}
    for idx, attn in enumerate(get_attention_modules(model)):
        proj = get_output_projection(attn)
        if proj is not None:
            out[f"layer{idx}.o_proj"] = proj
        for qkv_name, qkv in (get_qkv_projections(attn) or {}).items():
            if isinstance(qkv, nn.Linear):
                out[f"layer{idx}.{qkv_name}"] = qkv
    return out


def sparsity_report(model: nn.Module, layers: dict | None = None) -> dict:
    """Zero fraction over the targeted layers and over the whole model."""
    tot = zeros = 0
    for p in model.parameters():
        tot += p.numel()
        zeros += int((p == 0).sum().item())
    out = {"model_params": tot, "model_zeros": zeros,
           "model_zero_frac": zeros / max(tot, 1)}
    if layers:
        lt = lz = 0
        for lyr in layers.values():
            lt += lyr.weight.numel()
            lz += int((lyr.weight == 0).sum().item())
        out.update(target_params=lt, target_zeros=lz,
                   target_zero_frac=lz / max(lt, 1))
    return out


@torch.no_grad()
def prune_model(model: nn.Module, layers: dict, calib_batches, method: str,
                sparsity: float, device=None, log=print) -> dict:
    """Prune ``layers`` with ``method`` at ``sparsity``, one forward pass.

    Both methods need the *input* to each target layer, so every target gets a
    forward hook and the calibration set is pushed through the model once. The
    reference implementations walk block by block to keep activations off the
    GPU for a 70B model; at ViT-B/16 and BERT-base scale the whole-model pass
    is simpler and measures the same thing.
    """
    assert method in ("sparsegpt", "wanda"), method
    cls = SparseGPTPruner if method == "sparsegpt" else WandaPruner
    pruners = {name: cls(lyr) for name, lyr in layers.items()}

    handles = []
    for name, lyr in layers.items():
        def hook(_mod, inp, _out, _name=name):
            pruners[_name].add_batch(inp[0].detach())
        handles.append(lyr.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    n = 0
    for batch in calib_batches:
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            model(**batch)
            n += len(next(iter(batch.values())))
        else:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device) if device is not None else x
            model(x)
            n += len(x)
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    log(f"  calibrated {method} on {n} samples over {len(layers)} layers")

    zeroed = 0
    for name, pr in pruners.items():
        zeroed += pr.prune(sparsity)
    log(f"  zeroed {zeroed} weights at sparsity {sparsity}")

    rep = sparsity_report(model, layers)
    rep.update(method=method, sparsity=sparsity, calib_samples=n,
               layers=len(layers), zeroed=zeroed)
    return rep
