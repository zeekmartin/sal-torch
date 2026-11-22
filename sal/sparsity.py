"""One-shot unstructured and 2:4 pruning: SparseGPT and Wanda.

The product's own measurement says structure is the expensive axis, and until
v0.7.1 the package shipped only the expensive one. Removing 50% of a
ViT-B/16's attention weights without a structure constraint costs **1.23pp**
one-shot; removing the same parameters as whole heads costs **68.88pp**, both
from the same pretrained 78.44% checkpoint with zero epochs on either side
(exp33). After two epochs of recovery at a matched count of removed
parameters, every unstructured cell lands within **0.22pp** of its matched
control where structured head pruning costs **3.91pp**.

So this module exists to let a caller act on that, and :func:`prune_unstructured`
and :func:`prune_2of4` are its product surface.

**No speedup is promised, and the difference between the three structure
classes is the whole point.**

*Unstructured* zeros are scattered inside the weight matrices. The parameters
are **masked, not removed**, so a dense kernel does exactly as much arithmetic
as before: there is **no speedup to report**, by construction, and these cells
are compared on accuracy at a matched count of zeroed parameters and never on
latency.

*Semi-structured 2:4* is the pattern NVIDIA's sparse tensor cores accelerate,
so unlike unstructured it **can** be made faster -- on Ampere or newer, with a
kernel that exploits it. This project could not measure that: its hardware is
MI250X and ROCm has no equivalent path, so 2:4 speedup is recorded as
**unmeasurable rather than as none**. Those are different claims.

*Structured* removal (:func:`sal.slice_heads`, FFN neuron slicing) actually
takes weights out, so size and latency move. That is where this project's
measured speedups come from.

Attribution
-----------
Both algorithms are reimplemented here, following the reference
implementations closely enough that a layer-level equivalence test passes
(``tests/test_exp33_sparse_equivalence.py``: identical masks, weights
agreeing to 1e-5 for SparseGPT and 1e-6 for Wanda). They are vendored rather
than imported so that no third-party code runs inside this package and so
what ships is reviewable in one place. **The full licence texts, the upstream
commits and their digests are in ``THIRD_PARTY_LICENSES``**, which ships in
both the sdist and the wheel.

**SparseGPT** -- Frantar & Alistarh, "SparseGPT: Massive Language Models Can
Be Accurately Pruned in One-Shot" (ICML 2023).
Reference: https://github.com/IST-DASLab/sparsegpt, **Apache License 2.0**.
:class:`SparseGPTPruner` follows ``sparsegpt.py::SparseGPT`` -- the same
Hessian accumulation and rescaling, ``percdamp`` 0.01 damping on the
diagonal, the Cholesky / ``cholesky_inverse`` / upper-Cholesky route to
``Hinv``, a block size of 128, per-block mask selection by
``w**2 / diag(Hinv)**2`` against a per-block threshold, and the same
two-level error propagation (within the block column by column, then once
onto the columns after it).

**Wanda** -- Sun, Liu, Bair & Kolter, "A Simple and Effective Pruning
Approach for Large Language Models" (ICLR 2024).
Reference: https://github.com/locuslab/wanda, **MIT License**,
Copyright (c) 2023 CMU Locus Lab.
:class:`WandaPruner` follows ``image_classifiers/layerwrapper.py::WrappedLayer``
and ``image_classifiers/prune_utils.py::compute_mask`` -- the metric is
``|W| * sqrt(scaler_row)`` with ``scaler_row`` the running sum of per-batch
mean squared column norms, and the comparison group is **per output row**,
which is the paper's default.

Neither reference needs a custom kernel: the whole path is torch ops. The
ROCm-sensitive ones are ``torch.linalg.cholesky``, ``torch.cholesky_inverse``,
``torch.sort`` and ``torch.topk``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["prune_unstructured", "prune_2of4", "SparsityError",
           "SCOPES", "SparseGPTPruner", "WandaPruner",
           "prune_model", "linear_layers",
           "attention_linear_layers", "sparsity_report"]

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
              percdamp: float = PERCDAMP, prunen: int = 0,
              prunem: int = 0) -> int:
        """Prune and write the weights back. Returns the number zeroed.

        With ``prunen``/``prunem`` set (2:4 is ``prunen=2, prunem=4``) the
        reference builds the mask **inside** the column loop, in groups of
        ``prunem``, from the *already error-corrected* ``W1`` rather than from
        a metric computed up front. That ordering is the method, so it is
        reproduced rather than simplified. ``sparsity`` is then ignored,
        because n:m fixes the fraction by construction.
        """
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

            if prunen == 0:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if prunen != 0 and i % prunem == 0:
                    tmp = (W1[:, i:(i + prunem)] ** 2
                           / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2)
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1,
                                          largest=False)[1], True)
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

    def prune(self, sparsity: float, prunen: int = 0,
              prunem: int = 0) -> int:
        """Mask the lowest-metric weights. Returns the number zeroed.

        With ``prunen``/``prunem`` the comparison group is each run of
        ``prunem`` consecutive columns within a row, and ``prunen`` of them
        are zeroed -- the n:m branch of ``lib/prune.py::prune_wanda``.
        ``sparsity`` is then ignored, since n:m fixes the fraction.
        """
        W = self.layer.weight.data
        metric = torch.abs(W).float() * torch.sqrt(
            self.scaler_row.reshape((1, -1)))
        mask = torch.zeros_like(metric, dtype=torch.bool)
        if prunen != 0:
            for ii in range(0, metric.shape[1], prunem):
                tmp = metric[:, ii:(ii + prunem)].float()
                mask.scatter_(
                    1, ii + torch.topk(tmp, prunen, dim=1,
                                       largest=False)[1], True)
        else:
            # Per-output-row comparison group: each row loses the same fraction.
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

def _prune_in_groups(model, layers, calib_batches, method, sparsity,
                     device, log, prunen, prunem, group_size):
    """``prune_model`` over groups of targets, one calibration pass each.

    Buffers each group's pruned weights on CPU and restores the originals
    before the next pass, so every pass sees the dense model. Writes the
    buffered weights back once at the end.
    """
    names = list(layers)
    groups = [names[i:i + group_size]
              for i in range(0, len(names), group_size)]
    log("  grouping: %d target(s) in %d group(s) of up to %d, one "
        "calibration pass each; activations stay those of the DENSE model"
        % (len(names), len(groups), group_size))

    pruned_weights = {}
    # Only the ADDITIVE keys are accumulated. The per-group reports also
    # carry fractions and configuration values, and summing those is how
    # job 23839962 came to claim a zero fraction of 4.000012: eight
    # groups at 0.500001 each. It had pruned correctly and recovered
    # correctly -- the post-recovery measurement read 0.500001 -- and
    # then failed its own drift guard against a number that cannot
    # exist, since a fraction lies in [0, 1]. `sparsity` read 4.0 in the
    # same message for the same reason.
    _ADDITIVE = ("target_params", "target_zeros", "zeroed")
    report = {}
    totals = {k: 0 for k in _ADDITIVE}
    last = {}
    for gi, group in enumerate(groups):
        sub = {n: layers[n] for n in group}
        # The originals, so the model is dense again for the next pass.
        originals = {n: sub[n].weight.data.detach().clone() for n in group}
        r = prune_model(model, sub, calib_batches, method, sparsity,
                        device=device, log=(lambda *a, **k: None),
                        prunen=prunen, prunem=prunem)
        for n in group:
            pruned_weights[n] = sub[n].weight.data.detach().to("cpu").clone()
            sub[n].weight.data.copy_(originals[n])
        del originals
        if device is not None and getattr(device, "type", device) == "cuda":
            torch.cuda.empty_cache()
        r = r or {}
        last = r
        for k in _ADDITIVE:
            v = r.get(k)
            if isinstance(v, (int, float)):
                totals[k] += v
        log("    group %d/%d: %d layer(s) pruned" % (gi + 1, len(groups),
                                                     len(group)))

    with torch.no_grad():
        for n, w in pruned_weights.items():
            layers[n].weight.data.copy_(w.to(layers[n].weight.device))
    log("  grouping: wrote back %d pruned weight tensor(s)"
        % len(pruned_weights))

    # The report describes the model that now exists, so it is MEASURED
    # on it rather than arithmetically combined from the passes. Every
    # group saw a dense model with only its own targets pruned, so no
    # per-group whole-model figure describes the end state and none of
    # them can be made to by addition.
    report = sparsity_report(model, layers)
    report.update(method=last.get("method"),
                  sparsity=last.get("sparsity"),
                  calib_samples=last.get("calib_samples"),
                  nm=last.get("nm"),
                  layers=len(layers),
                  zeroed=totals["zeroed"],
                  grouped=dict(groups=len(groups), group_size=group_size,
                               target_params_summed=totals["target_params"],
                               target_zeros_summed=totals["target_zeros"]))
    # The summed counts and the measured ones are two routes to the same
    # quantity, so a disagreement means a target was pruned twice or not
    # at all. Cheap, and it is the check that would have caught this.
    if report.get("target_params") != totals["target_params"]:
        raise RuntimeError(
            "grouping covered %d target parameter(s) across %d group(s) "
            "but the pruned model holds %d. Every target must appear in "
            "exactly one group."
            % (totals["target_params"], len(groups),
               report.get("target_params")))
    return report


def prune_model(model: nn.Module, layers: dict, calib_batches, method: str,
                sparsity: float, device=None, log=print, prunen: int = 0,
                prunem: int = 0, group_size: int | None = None) -> dict:
    """Prune ``layers`` with ``method`` at ``sparsity``.

    Both methods need the *input* to each target layer, so every target gets a
    forward hook and the calibration set is pushed through the model once. The
    reference implementations walk block by block to keep activations off the
    GPU for a 70B model; at ViT-B/16 and BERT-base scale the whole-model pass
    is simpler and measures the same thing.

    ``group_size`` bounds the memory of that pass without changing what it
    computes. Every pruner holds an ``in_features**2`` fp32 Hessian, so
    keeping them all alive costs 7.9 GiB on OPT-1.3B and 42.0 GiB on
    OPT-6.7B, which with fp32 weights exceeds a 64 GiB GCD. With a group
    size, the targets are pruned in groups of that many, one calibration
    pass each, and only one group's pruners exist at a time.

    **Grouping does not change the result.** A group's pruned weights are
    buffered and the originals restored before the next pass, so every
    pass sees the DENSE model exactly as the single-pass version does;
    the buffered weights are written back at the end. This matters because
    the reference implementations are sequential -- a layer there sees
    activations from a model whose earlier layers are already pruned --
    and every figure this campaign has published was measured the other
    way. ``tests/test_sparse_grouping.py`` pins that grouped and
    whole-model passes produce bit-identical masks and weights.
    """
    assert method in ("sparsegpt", "wanda"), method
    if group_size:
        return _prune_in_groups(model, layers, calib_batches, method,
                                sparsity, device, log, prunen, prunem,
                                group_size)
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
        zeroed += pr.prune(sparsity, prunen=prunen, prunem=prunem)
    shape = f"{prunen}:{prunem}" if prunen else f"sparsity {sparsity}"
    log(f"  zeroed {zeroed} weights at {shape}")

    rep = sparsity_report(model, layers)
    rep.update(method=method, sparsity=sparsity, calib_samples=n,
               layers=len(layers), zeroed=zeroed,
               nm=(f"{prunen}:{prunem}" if prunen else None))
    return rep


# ======================================================= the product surface
#: What `scope` selects. "attention" is the scope every published figure in
#: this project was measured at; "whole" removes from every Linear and its
#: costs differ, including its INT4 interaction, which runs the opposite way
#: to the fp32 ordering on 4 of 4 pairings.
SCOPES = ("attention", "whole")


class SparsityError(Exception):
    """Raised when a pruning request cannot be honoured honestly."""


def _targets(model: nn.Module, scope: str) -> dict:
    if scope not in SCOPES:
        raise SparsityError("scope must be one of %s, got %r"
                            % (SCOPES, scope))
    layers = (attention_linear_layers(model) if scope == "attention"
              else linear_layers(model))
    if not layers:
        raise SparsityError(
            "no %s Linear layers found in this model. Nothing was pruned, "
            "which is reported rather than returning a report of zeros."
            % scope)
    return layers


def prune_unstructured(model: nn.Module, calib_batches, *, ratio: float = 0.50,
                       method: str = "wanda", scope: str = "attention",
                       group_size=None, device=None, log=print) -> dict:
    """Zero ``ratio`` of the weights in ``scope``, one-shot, and report it.

    ``method="wanda"`` by default, and the reason is measured rather than
    aesthetic. SparseGPT's error compensation earns its keep only as the
    constraint tightens: it **loses** to Wanda by 0.39pp one-shot at
    attention-P33, ties at attention-P50 (0.06pp), and wins by 2.35pp only at
    2:4. After a recovery pass every pairing is inside the seed spread. It is
    also 2.7 to 24.3x dearer to run over twelve matched cells. For the
    recommended path -- prune, then recover -- the cheaper method is the
    right default.

    **This does not make the model smaller or faster.** The zeros are
    scattered inside the weight matrices, the shapes do not change, and a
    dense kernel does exactly as much arithmetic as before. What it buys is
    accuracy retention at a matched count of removed parameters: within
    0.22pp of a matched control after two epochs, where removing the same
    parameters as whole heads costs 3.91pp.

    Pair it with ``recover(..., preserve_sparsity=True)``, which is the
    default: the zero pattern is captured, defended by a gradient hook, and
    re-applied once at the end, with the drift measured **before** that
    re-apply so the re-apply cannot hide a failure.

    ``group_size`` bounds the memory of the calibration pass without changing
    what it computes. Every SparseGPT pruner holds an ``in_features**2`` fp32
    Hessian, so keeping them all alive costs 7.9 GiB on a 1.3B decoder and
    42.0 GiB on a 6.7B one, which with fp32 weights exceeds a 64 GiB
    accelerator.
    """
    if not 0.0 < ratio < 1.0:
        raise SparsityError("ratio must be in (0, 1), got %r" % (ratio,))
    layers = _targets(model, scope)
    rec = prune_model(model, layers, calib_batches, method, ratio,
                      device=device, log=log, group_size=group_size)
    rec.update({
        "api": "sal.prune_unstructured",
        "scope": scope,
        "structure": "unstructured",
        "speedup": "none by construction: the parameters are masked, not "
                   "removed, so a dense kernel does the same arithmetic",
        "pair_with": "recover(..., preserve_sparsity=True), which is the "
                     "default",
    })
    return rec


def prune_2of4(model: nn.Module, calib_batches, *,
               method: str = "sparsegpt", scope: str = "attention",
               group_size=None, device=None, log=print) -> dict:
    """Zero two of every four consecutive weights in a row, and report it.

    ``method="sparsegpt"`` by default here and ``"wanda"`` in
    :func:`prune_unstructured`, and the asymmetry is the measurement: 2:4 is
    the one cell where SparseGPT's error compensation wins, by **2.35pp**
    one-shot. That is the reading that makes mechanistic sense -- error
    compensation matters when the mask is forced into a pattern the weights
    did not choose.

    The sparsity is fixed by the format at one half; there is no ``ratio``.

    **The speedup is UNMEASURABLE here, which is not the same as none.** The
    2:4 pattern is what NVIDIA's sparse tensor cores accelerate, so unlike
    unstructured sparsity it can be made faster on Ampere or newer with a
    kernel that exploits it. This project's hardware is MI250X and ROCm has
    no equivalent path, so no speedup figure exists either way, and the
    record says unmeasurable rather than quietly reporting 1.00x.
    """
    layers = _targets(model, scope)
    rec = prune_model(model, layers, calib_batches, method, 0.5,
                      device=device, log=log, prunen=2, prunem=4,
                      group_size=group_size)
    rec.update({
        "api": "sal.prune_2of4",
        "scope": scope,
        "structure": "2:4 semi-structured",
        "sparsity_fixed_by_format": 0.5,
        "speedup": "UNMEASURABLE on this hardware, which is not none: 2:4 "
                   "can be accelerated on Ampere or newer with a kernel "
                   "that reads it, and ROCm has no equivalent path",
        "pair_with": "recover(..., preserve_sparsity=True), which is the "
                     "default",
    })
    return rec
