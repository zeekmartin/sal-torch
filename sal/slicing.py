"""Head slicing — physically remove attention heads from the weights.

Masking a head with a forward hook makes it *behave* as if it were gone. It does
not make the model smaller: the weights are still there, the matmuls are still
full width, and the checkpoint is the same size on disk. Slicing is the step that
turns a measurement into a saving.

    from sal import slice_heads

    small = slice_heads(model, heads_to_remove=[(0, 3), (0, 7), (1, 2), ...])

The returned model:

  * has narrower Q/K/V projections (rows for removed heads are gone) and a
    narrower output projection (the matching input columns are gone),
  * carries updated head bookkeeping on every attention module and on
    ``config.num_attention_heads``,
  * runs **without any hooks and without sal-torch installed** — it is an
    ordinary model of its own architecture.

What it will not do
-------------------
**Uniform removal only.** Every layer must give up the same number of heads.
Architectures store one head count, and per-layer variation cannot survive a
``save_pretrained``/``from_pretrained`` round trip. A model that only works
in-memory is not a compressed model, so this raises instead.

**Multi-head attention only.** Under grouped-query attention the query heads are
tied to shared key/value heads, and removing query heads without removing whole
groups silently corrupts the mapping. That needs its own design; until then it
raises rather than producing a model that is quietly wrong.

Round-tripping through ``save_pretrained``
------------------------------------------
The sliced model always runs standalone in memory. Whether it can be *reloaded*
from a config depends on the architecture: llama-style models that carry an
explicit ``head_dim`` rebuild correctly, while GPT-2 and BERT derive attention
width from ``hidden_size`` and will rebuild at full width. Use
:func:`verify_roundtrip` to find out for a given model rather than assuming, and
see :meth:`sal.pipeline.CompressionPipeline.export`, which checks for you.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

from sal import arch_support

logger = logging.getLogger(__name__)

# Attributes holding "how many heads does this module have".
_HEAD_COUNT_ATTRS = ("num_heads", "n_heads", "num_attention_heads", "n_head", "nh")
# Attributes holding "how wide is the concatenated per-head representation".
_ATTN_WIDTH_ATTRS = ("split_size", "all_head_size")


class SlicingError(Exception):
    """Raised when a model cannot be sliced safely."""


def _is_conv1d(mod) -> bool:
    """GPT-2-style Conv1D: weight is [in, out], the transpose of nn.Linear."""
    return type(mod).__name__ == "Conv1D"


def _row_index(heads, head_dim: int, offset: int = 0) -> list:
    """Flat weight indices covering ``heads``, each ``head_dim`` wide."""
    idx = []
    for h in heads:
        idx.extend(range(offset + h * head_dim, offset + (h + 1) * head_dim))
    return idx


def _slice_out(mod, keep_idx: list):
    """Keep only ``keep_idx`` along the module's OUTPUT dimension, in place."""
    index = torch.tensor(keep_idx, dtype=torch.long, device=mod.weight.device)
    if _is_conv1d(mod):                       # weight [in, out]
        mod.weight = nn.Parameter(mod.weight.data.index_select(1, index),
                                  requires_grad=mod.weight.requires_grad)
        mod.nf = len(keep_idx)
    else:                                     # nn.Linear weight [out, in]
        mod.weight = nn.Parameter(mod.weight.data.index_select(0, index),
                                  requires_grad=mod.weight.requires_grad)
        mod.out_features = len(keep_idx)
    if getattr(mod, "bias", None) is not None:
        mod.bias = nn.Parameter(mod.bias.data.index_select(0, index),
                                requires_grad=mod.bias.requires_grad)


def _slice_in(mod, keep_idx: list):
    """Keep only ``keep_idx`` along the module's INPUT dimension, in place.

    The bias is per output feature and mixes every head, so it is left alone.
    """
    index = torch.tensor(keep_idx, dtype=torch.long, device=mod.weight.device)
    if _is_conv1d(mod):                       # weight [in, out]
        mod.weight = nn.Parameter(mod.weight.data.index_select(0, index),
                                  requires_grad=mod.weight.requires_grad)
    else:                                     # nn.Linear weight [out, in]
        mod.weight = nn.Parameter(mod.weight.data.index_select(1, index),
                                  requires_grad=mod.weight.requires_grad)
        mod.in_features = len(keep_idx)


def _retarget(mod, old_n: int, new_n: int, head_dim: int):
    """Rewrite one module's head bookkeeping after slicing.

    Only attributes that already exist are touched, and only with their known
    meaning — head counts become ``new_n``, concatenated-width attributes become
    ``new_n * head_dim``. ``head_dim`` itself never changes: slicing removes
    whole heads, it does not reshape them.
    """
    for name in _HEAD_COUNT_ATTRS:
        if hasattr(mod, name) and isinstance(getattr(mod, name), int):
            setattr(mod, name, new_n)
    # Only meaningful for MHA; grouped-query models are rejected up front.
    if isinstance(getattr(mod, "num_key_value_heads", None), int):
        mod.num_key_value_heads = new_n
    if isinstance(getattr(mod, "num_key_value_groups", None), int):
        mod.num_key_value_groups = 1
    for name in _ATTN_WIDTH_ATTRS:
        if hasattr(mod, name) and isinstance(getattr(mod, name), int):
            setattr(mod, name, new_n * head_dim)
    # DistilBERT derives its per-head width as `dim // n_heads` at forward time,
    # so `dim` here means the attention width, not the residual width.
    if (hasattr(mod, "dim") and hasattr(mod, "n_heads")
            and not hasattr(mod, "head_dim") and isinstance(mod.dim, int)):
        mod.dim = new_n * head_dim


def _normalize(heads_to_remove) -> dict:
    by_layer: dict = {}
    for item in heads_to_remove:
        layer, head = int(item[0]), int(item[1])
        by_layer.setdefault(layer, set()).add(head)
    return {k: sorted(v) for k, v in by_layer.items()}


def _check_uniform(by_layer: dict, num_layers: int, num_heads: int) -> int:
    counts = [len(by_layer.get(i, [])) for i in range(num_layers)]
    unknown = [i for i in by_layer if i < 0 or i >= num_layers]
    if unknown:
        raise SlicingError(f"Layer index out of range: {unknown} (model has {num_layers} layers).")
    bad_heads = {i: [h for h in hs if h < 0 or h >= num_heads] for i, hs in by_layer.items()}
    bad_heads = {i: hs for i, hs in bad_heads.items() if hs}
    if bad_heads:
        raise SlicingError(f"Head index out of range: {bad_heads} "
                           f"(each layer has {num_heads} heads).")
    k = counts[0]
    if any(c != k for c in counts):
        per_layer = {i: c for i, c in enumerate(counts)}
        raise SlicingError(
            "Head slicing needs the same number of heads removed from every layer, "
            f"got {per_layer}. Architectures store a single head count, so uneven "
            "layers cannot be saved and reloaded. Balance the selection first — "
            "PlasticityScanner.recommend() ranks heads within a layer, so take the "
            "same number from each.")
    if k == 0:
        raise SlicingError("Nothing to remove.")
    if k >= num_heads:
        raise SlicingError(f"Cannot remove all {num_heads} heads from a layer.")
    return k


def slice_heads(model: nn.Module, heads_to_remove, verify_input=None,
                atol: float = 1e-5, rtol: float = 1e-5, inplace: bool = False) -> nn.Module:
    """Physically remove ``heads_to_remove`` from ``model``'s attention weights.

    ``heads_to_remove`` is an iterable of ``(layer, head)`` pairs. The same number
    must be removed from every layer — see the module docstring for why.

    Pass ``verify_input`` (a batch dict or tensor the model accepts) to check the
    sliced model against the equivalent masked model before returning; a mismatch
    beyond ``atol`` raises rather than handing back a quietly wrong model.

    Returns a new model unless ``inplace=True``.
    """
    by_layer = _normalize(heads_to_remove)
    attn_mods = arch_support.get_attention_modules(model)
    if not attn_mods:
        raise SlicingError("No attention modules found; cannot slice this model.")
    num_layers = len(attn_mods)

    from sal.fi import _infer_num_heads
    old_n = _infer_num_heads(model)
    k = _check_uniform(by_layer, num_layers, old_n)
    new_n = old_n - k

    cfg = getattr(model, "config", None)
    kv = getattr(cfg, "num_key_value_heads", None) if cfg is not None else None
    if kv is not None and kv != old_n:
        raise SlicingError(
            f"This model uses grouped-query attention ({old_n} query heads over {kv} "
            "key/value heads). Removing query heads without removing whole KV groups "
            "would corrupt the query-to-group mapping, so slicing refuses rather than "
            "producing a model that is subtly wrong. Head *masking* still works.")

    reference = None
    if verify_input is not None:
        reference = _masked_reference(model, by_layer, old_n, verify_input)

    target = model if inplace else copy.deepcopy(model)
    target_mods = arch_support.get_attention_modules(target)
    head_dim = None

    for li, attn in enumerate(target_mods):
        remove = by_layer.get(li, [])
        keep = [h for h in range(old_n) if h not in set(remove)]

        out_proj = arch_support.get_output_projection(attn)
        if out_proj is None:
            raise SlicingError(f"No attention output projection at layer {li}.")
        # Width of the concatenated per-head representation, read off the
        # projection that consumes it.
        in_width = (out_proj.weight.shape[0] if _is_conv1d(out_proj)
                    else out_proj.weight.shape[1])
        hd, rem = divmod(in_width, old_n)
        if rem:
            raise SlicingError(f"Layer {li}: attention width {in_width} is not divisible "
                               f"by {old_n} heads.")
        head_dim = hd if head_dim is None else head_dim
        if hd != head_dim:
            raise SlicingError("Layers disagree on head_dim; cannot slice uniformly.")

        _slice_in(out_proj, _row_index(keep, hd))

        qkv = arch_support.get_qkv_projections(attn)
        if qkv is None:
            raise SlicingError(f"Layer {li}: could not locate Q/K/V projections.")
        if qkv["mode"] == "separate":
            for key in ("q", "k", "v"):
                _slice_out(qkv[key], _row_index(keep, hd))
        else:
            fused = qkv["qkv"]
            out_dim = (fused.weight.shape[1] if _is_conv1d(fused) else fused.weight.shape[0])
            block, rem = divmod(out_dim, 3)
            if rem or block != old_n * hd:
                raise SlicingError(
                    f"Layer {li}: fused Q|K|V projection has width {out_dim}, which is not "
                    f"three equal blocks of {old_n * hd}. Fused layouts with unequal "
                    "Q/K/V widths are not supported.")
            keep_idx = []
            for b in range(3):
                keep_idx.extend(_row_index(keep, hd, offset=b * block))
            _slice_out(fused, keep_idx)

        _retarget(attn, old_n, new_n, hd)
        sub = getattr(attn, "self", None)
        if sub is not None:
            _retarget(sub, old_n, new_n, hd)

    # The *target's* config, not the source's — writing to `cfg` here would mutate
    # the caller's model even with inplace=False, and leave the copy unchanged.
    target_cfg = getattr(target, "config", None)
    if target_cfg is not None:
        try:
            target_cfg.num_attention_heads = new_n
            if getattr(target_cfg, "num_key_value_heads", None) is not None:
                target_cfg.num_key_value_heads = new_n
        except Exception as e:  # noqa: BLE001 — some configs are read-only namespaces
            logger.warning(f"Could not update config head count: {e}")

    logger.info(f"Sliced {k} head(s) from each of {num_layers} layers: "
                f"{old_n} -> {new_n} heads per layer.")

    if reference is not None:
        _verify_against(target, reference, verify_input, atol, rtol)
    return target


# ------------------------------------------------------------------ verification
def _masked_reference(model, by_layer: dict, num_heads: int, batch):
    """Output of ``model`` with the same heads zeroed by hook, for comparison."""
    projs = arch_support.get_output_projections(model)
    handles = []

    def make(heads):
        def hook(mod, inputs):
            x = inputs[0]
            width = x.shape[-1]
            hd = width // num_heads
            x = x.clone()
            for h in heads:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(inputs[1:])
        return hook

    try:
        for li, proj in enumerate(projs):
            heads = by_layer.get(li)
            if heads:
                handles.append(proj.register_forward_pre_hook(make(heads)))
        return _forward_logits(model, batch).detach().clone()
    finally:
        for h in handles:
            h.remove()


def _forward_logits(model, batch):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(**batch) if isinstance(batch, dict) else model(batch)
        return out.logits if hasattr(out, "logits") else out
    finally:
        model.train(was_training)


def _verify_against(sliced, reference, batch, atol: float, rtol: float = 1e-5):
    """Compare against the masked reference with ``torch.allclose`` semantics.

    A purely absolute tolerance cannot work across model scales. Slicing narrows
    the reductions feeding every matmul (for GPT-2, 768 terms become 576), so the
    result differs from the masked model by float reassociation alone — about
    9e-5 on logits of magnitude ~40, which is ~2e-6 relative. A genuine
    bookkeeping bug is not subtle by comparison: mis-set head counts reshape the
    tensor and shift outputs by O(0.1) relative. Scaling the threshold by the
    reference magnitude separates the two; a flat 1e-5 would just fail on every
    real model.
    """
    got = _forward_logits(sliced, batch)
    if got.shape != reference.shape:
        raise SlicingError(f"Sliced model output shape {tuple(got.shape)} != masked "
                           f"reference {tuple(reference.shape)}.")
    diff = (got.float() - reference.float()).abs().max().item()
    scale = reference.float().abs().max().item()
    threshold = atol + rtol * scale
    if diff > threshold:
        raise SlicingError(
            f"Sliced model does not match the masked model (max abs diff {diff:.3g} > "
            f"{threshold:.3g}, reference scale {scale:.3g}). The head bookkeeping for "
            "this architecture is probably not handled; please report the model type.")
    logger.info(f"Slicing verified against masked reference "
                f"(max abs diff {diff:.3g}, threshold {threshold:.3g}).")


def verify_slicing(model, sliced, heads_to_remove, batch, atol: float = 1e-5) -> float:
    """Max absolute difference between the sliced model and the masked original.

    Returns the difference rather than raising, so callers can report it.
    """
    by_layer = _normalize(heads_to_remove)
    from sal.fi import _infer_num_heads
    reference = _masked_reference(model, by_layer, _infer_num_heads(model), batch)
    got = _forward_logits(sliced, batch)
    return float((got.float() - reference.float()).abs().max().item())


def verify_roundtrip(sliced, batch, tmpdir: Optional[str] = None) -> bool:
    """Can this sliced model be saved and reloaded from its own config?

    Architectures that derive attention width from ``hidden_size`` (GPT-2, BERT)
    rebuild at full width and cannot round-trip; ones that carry an explicit
    ``head_dim`` can. Returns False rather than raising.
    """
    import tempfile

    if not hasattr(sliced, "save_pretrained"):
        return False
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        try:
            sliced.save_pretrained(d)
            # The model's own class, not an Auto* guess at its head type.
            reloaded = type(sliced).from_pretrained(d)
        except Exception as e:  # noqa: BLE001 — the point of the check
            logger.info(f"Round-trip unavailable for this architecture: {e}")
            return False
        try:
            a = _forward_logits(sliced, batch)
            b = _forward_logits(reloaded, batch)
            return bool(a.shape == b.shape
                        and (a.float() - b.float()).abs().max().item() < 1e-4)
        except Exception as e:  # noqa: BLE001
            logger.info(f"Round-trip forward failed: {e}")
            return False


def head_savings(model, heads_removed_per_layer: int) -> dict:
    """Parameter count removed by slicing, before actually doing it."""
    from sal.fi import _infer_num_heads
    nh = _infer_num_heads(model)
    attn_mods = arch_support.get_attention_modules(model)
    total = sum(p.numel() for p in model.parameters())
    removed = 0
    for attn in attn_mods:
        out_proj = arch_support.get_output_projection(attn)
        if out_proj is None:
            continue
        width = (out_proj.weight.shape[0] if _is_conv1d(out_proj) else out_proj.weight.shape[1])
        hd = width // nh
        hidden = (out_proj.weight.shape[1] if _is_conv1d(out_proj) else out_proj.weight.shape[0])
        # Q, K, V lose rows (+bias); O loses columns.
        removed += heads_removed_per_layer * hd * (3 * hidden + hidden)
    return {"total_params": total, "removed_params": removed,
            "remaining_params": total - removed,
            "fraction_removed": removed / max(total, 1)}
