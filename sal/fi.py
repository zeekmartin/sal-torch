"""Fragility Index (FI) computation.

FI is a structural fragility score for a model's attention graph, in [0, 1].
It is the **fraction of edges that lack triangle support** — edges connecting
two heads that share no common neighbour. Such edges have no redundant
pathway, so the function they carry has no backup.

  * Low FI  → heavily triangulated graph, lots of redundant pathways → robust.
  * High FI → many unsupported edges → fragile.

The graph is built from per-head activation signatures: each head's signature is
captured at the input to the attention output projection (where feature
dimensions map onto heads), reduced to a per-head feature vector, and compared
to every other head via Pearson similarity. Edges are kept at a fixed target
density.
"""
from __future__ import annotations
import logging
from enum import Enum
from typing import Optional
import numpy as np
import torch, torch.nn as nn

logger = logging.getLogger(__name__)

# Fraction of head-pairs kept as edges when thresholding the similarity matrix.
DEFAULT_EDGE_DENSITY = 0.10


class LayerClass(str, Enum):
    IMMUNE = "IMMUNE"
    BUFFER = "BUFFER"
    CRITICAL = "CRITICAL"


def extract_activation_graph(model: nn.Module, probe_dataset, num_samples: int = 500,
                             batch_size: int = 16, density: float = DEFAULT_EDGE_DENSITY) -> np.ndarray:
    """Build the binary head-adjacency graph from probe activations."""
    model.eval()
    device = next(model.parameters()).device
    out_projs = _find_output_projections(model)
    num_heads = _infer_num_heads(model)
    nl = len(out_projs)
    captures: list = [None] * nl
    per_layer: list[list] = [[] for _ in range(nl)]

    def make_hook(idx):
        def fn(mod, inputs):
            captures[idx] = inputs[0].detach()
        return fn

    hooks = [proj.register_forward_pre_hook(make_hook(i)) for i, proj in enumerate(out_projs)]
    try:
        n = 0
        with torch.no_grad():
            for batch in _iter_data(probe_dataset, batch_size):
                batch = _to_dev(batch, device)
                attn = batch.get("attention_mask") if isinstance(batch, dict) else None
                model(**batch) if isinstance(batch, dict) else model(batch)
                if attn is not None:
                    mask = attn.unsqueeze(-1).float()
                    counts = mask.sum(dim=1).clamp(min=1)
                bs = 0
                for i in range(nl):
                    x = captures[i]
                    if x is None:
                        continue
                    # Mask-weighted mean over the sequence, per sentence.
                    avg = (x * mask).sum(dim=1) / counts if attn is not None else x.mean(dim=1)
                    per_layer[i].append(avg.cpu())
                    captures[i] = None
                    bs = avg.shape[0]
                n += bs
                if n >= num_samples:
                    break
    finally:
        for h in hooks:
            h.remove()

    # Build one signature vector per head: [num_sentences, head_dim] flattened.
    sigs = []
    for i in range(nl):
        if not per_layer[i]:
            continue
        cat = torch.cat(per_layer[i], dim=0)[:num_samples]
        cat = cat.view(cat.shape[0], num_heads, -1)
        for h in range(num_heads):
            sigs.append(cat[:, h, :].reshape(-1).double().numpy())

    total = nl * num_heads
    if not sigs:
        return np.zeros((total, total), dtype=np.int8)

    X = np.stack(sigs)
    S = _pearson_similarity(X)
    thr = _threshold_for_density(S, density)
    adj = (np.abs(S) > thr).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj


def compute_fi(adjacency: np.ndarray) -> float:
    """Fragility Index: fraction of edges with zero triangle support.

    An edge (i, j) is fragile when heads i and j share no common neighbour,
    i.e. (A @ A)[i, j] == 0. Returns a value in [0, 1].
    """
    A = (np.asarray(adjacency) != 0).astype(np.int64)
    n = A.shape[0]
    if n < 2:
        return 0.0
    np.fill_diagonal(A, 0)
    common_neighbors = A @ A  # (A^2)[i,j] = number of shared neighbours of i, j
    triu = np.triu_indices(n, k=1)
    edges = A[triu] > 0
    total = int(edges.sum())
    if total == 0:
        return 1.0
    tri_support = common_neighbors[triu][edges]
    return float(np.sum(tri_support == 0) / total)


def layer_fi_deltas(model: nn.Module, adjacency: np.ndarray,
                    num_heads_per_layer: Optional[int] = None) -> dict[int, float]:
    """Relative FI change when each layer's heads are removed from the graph.

    This is the quantity both classifiers rank and threshold, returned on its
    own so it can be stored. Storing it is worth doing: it is a few hundred
    floats, and with it in hand any threshold or any ``k`` can be evaluated
    later without touching a GPU.

    The value is ``|FI_without_layer - FI| / FI``, so it is a **magnitude**: a
    layer whose removal makes the graph *less* fragile scores just as high as
    one that makes it more fragile. Returns ``{layer_index: delta}``, and all
    zeros when the graph's FI is zero (nothing to be relative to).
    """
    if num_heads_per_layer is None:
        num_heads_per_layer = _infer_num_heads(model)
    total = adjacency.shape[0]
    nl = total // num_heads_per_layer
    base_fi = compute_fi(adjacency)
    if base_fi < 1e-10:
        return {i: 0.0 for i in range(nl)}
    deltas = {}
    for li in range(nl):
        s, e = li * num_heads_per_layer, (li + 1) * num_heads_per_layer
        keep = np.ones(total, dtype=bool); keep[s:e] = False
        red_fi = compute_fi(adjacency[np.ix_(keep, keep)])
        deltas[li] = float(abs(red_fi - base_fi) / max(base_fi, 1e-10))
    return deltas


def classify_layers(model: nn.Module, adjacency: np.ndarray, num_heads_per_layer: Optional[int] = None,
                    immune_thr: float = 0.01, critical_thr: float = 0.25) -> dict[int, LayerClass]:
    """Classify each layer by how much removing its heads changes FI.

    IMMUNE   : relative FI change < ``immune_thr`` (default 1%)
    CRITICAL : relative FI change >= ``critical_thr`` (default 25%)
    BUFFER   : in between

    **``critical_thr`` was 0.05 and is now 0.25.** At 5% the classification
    saturates: on ViT-B/16 / ImageNet-1k it called 11 / 10 / 11 of 12 layers
    CRITICAL on three probes, which is not a classification. 5% of an FI of
    0.0709 is 3.6 edges out of 1030 — barely above the two-edge resolution of
    the graph itself, so almost every layer clears it. Sweeping the threshold
    on stored deltas, every value in (0.205, 0.295] put all three probes in a
    3-8 CRITICAL range, and 0.25 gave 7 / 6 / 5.

    That measurement is **one architecture**: 12 layers x 12 heads at FI 0.05-0.08.
    A model with a very different graph may well want a different number, and
    the deltas are not scale-free, so pass ``critical_thr`` explicitly rather
    than assuming either default transfers. If you want a classification that
    does not depend on picking a number at all, use
    :func:`classify_layers_by_rank` — across those three probes the *ordering*
    was stable (one layer most sensitive 3/3, another least 3/3) where no
    absolute threshold was.
    """
    deltas = layer_fi_deltas(model, adjacency, num_heads_per_layer)
    result = {}
    for li, delta in deltas.items():
        if delta < immune_thr:
            result[li] = LayerClass.IMMUNE
        elif delta >= critical_thr:
            result[li] = LayerClass.CRITICAL
        else:
            result[li] = LayerClass.BUFFER
    return result


def classify_layers_by_rank(fi_deltas: dict[int, float], k: int = 3) -> list[int]:
    """The ``k`` layers whose removal moves FI most, ranked, no threshold.

    Takes the mapping :func:`layer_fi_deltas` returns and gives back layer
    indices, most sensitive first. Ties break on the lower layer index, so the
    result is deterministic.

    **Why a rank and not a threshold.** The deltas are not scale-free: they
    depend on the model, the probe and the graph's edge density, so a number
    that classifies one model sensibly saturates on another (see
    ``critical_thr`` in :func:`classify_layers`). The *ordering* is the part
    that held across probes in the only place it has been measured — exp30b,
    three probes of the same ViT-B/16, where one layer ranked most sensitive
    3/3 and another least 3/3 while no fixed threshold put all three probes in
    a usable range. ``k`` is a budget the caller already has ("I can afford to
    protect three layers"), not a property of the model that has to be guessed.

    ``k`` is clamped to the number of layers; ``k <= 0`` returns ``[]``.
    """
    if k <= 0:
        return []
    ordered = sorted(fi_deltas.items(), key=lambda kv: (-kv[1], kv[0]))
    return [li for li, _ in ordered[:k]]


# ------------------------------------------------------------------ internals
def _pearson_similarity(X: np.ndarray) -> np.ndarray:
    """Row-wise Pearson similarity matrix (mean-center, L2-normalize, Gram)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    Xn = Xc / norms
    return Xn @ Xn.T


def _threshold_for_density(S: np.ndarray, density: float) -> float:
    """Similarity threshold that keeps ~``density`` of head-pairs as edges."""
    n = S.shape[0]
    triu = np.triu_indices(n, k=1)
    abs_vals = np.abs(S[triu])
    if abs_vals.size == 0:
        return 0.0
    q = max(0.0, min(1.0, 1.0 - density))
    return float(np.quantile(abs_vals, q))


def _find_output_projections(model):
    """Per-layer attention output projections (shared with the masker)."""
    from sal import arch_support
    return arch_support.get_output_projections(model)


def _infer_num_heads(model) -> int:
    cfg = getattr(model, 'config', None)
    if cfg is None:
        raise ValueError("No .config")
    for attr in ["num_attention_heads", "n_head", "n_heads"]:
        v = getattr(cfg, attr, None)
        if v is not None:
            return v
    raise ValueError("Cannot infer num_heads")


def _iter_data(dataset, bs):
    if hasattr(dataset, '__iter__'):
        yield from dataset
    else:
        from torch.utils.data import DataLoader
        yield from DataLoader(dataset, batch_size=bs, shuffle=False)


def _to_dev(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch.to(device) if isinstance(batch, torch.Tensor) else batch
