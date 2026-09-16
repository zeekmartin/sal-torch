"""Visualize what a model *sees*, before and after compression.

``sal.visualize`` charts numbers the scanners produced — fragility heatmaps,
plasticity bars, waterfalls. This module charts the model's own activations
instead: attention maps and feature maps for one concrete image, side by side
across two models.

The reason for a second module is that a number and a picture answer different
questions. "FI went from 0.31 to 0.28" is a claim about structure. "Here is the
same image through both encoders, and the compressed one has lost the object
boundary" is a claim anyone can check by looking. For an edge-robotics
conversation the second is the one that lands.

Needs matplotlib (``pip install sal-torch[reports]``). Every function returns
the matplotlib ``Figure`` so callers can adjust it before saving; passing
``save_path=`` writes it out as well.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sal import arch_support

logger = logging.getLogger(__name__)

__all__ = [
    "extract_attention_maps", "extract_feature_maps", "visualize_attention_maps",
    "visualize_feature_maps", "compare_feature_maps", "visualize_compression_impact",
]

_SAL_COLOR = "#2e7d32"       # green  — the SAL arm
_BASELINE_COLOR = "#757575"  # grey   — the baseline arm

# Most prefix tokens a ViT is plausibly carrying ahead of its patch grid: one
# CLS plus up to four registers, the DINOv2 layout. Dropping more than this to
# force a square would be inventing a layout, not recovering one — 23 tokens
# become a 4x4 grid if you are willing to discard seven of them, and that
# grid would be pure coincidence.
_MAX_PREFIX_TOKENS = 5


def _require_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no display needed
        import matplotlib.pyplot as plt
    except ImportError as e:    # pragma: no cover - exercised only without extras
        raise ImportError("Visualization needs matplotlib. "
                          "Install with: pip install sal-torch[reports]") from e
    return plt


def _as_batch(image: torch.Tensor) -> torch.Tensor:
    """Accept [C, H, W] or [B, C, H, W]; always return a batch of one."""
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() == 4:
        return image[:1]
    raise ValueError(
        f"Expected an image tensor of shape [C, H, W] or [B, C, H, W], "
        f"got {tuple(image.shape)}.")


def _run(model: nn.Module, image):
    """Forward one image through the model in eval mode, restoring its state."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    try:
        with torch.no_grad():
            if isinstance(image, dict):
                return model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in image.items()})
            return model(_as_batch(image).to(device))
    finally:
        model.train(was_training)


# ------------------------------------------------------------------ extraction
def extract_attention_maps(model: nn.Module, image, layer_idx: int = -1) -> np.ndarray:
    """Per-head attention for one image at one layer.

    Returns ``[num_heads, queries, keys]``. Requires a model that returns
    attention weights — for HF models load with
    ``attn_implementation="eager"``, or attentions come back empty and the plot
    silently shows nothing.
    """
    out = _run(model, image if isinstance(image, dict) else image)
    attns = getattr(out, "attentions", None)
    if attns is None and isinstance(out, dict):
        attns = out.get("attentions")
    if not attns:
        # Try again asking for them explicitly.
        kwargs = dict(image) if isinstance(image, dict) else None
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if kwargs is not None:
                    out = model(**kwargs, output_attentions=True)
                else:
                    out = model(_as_batch(image), output_attentions=True)
        except TypeError as e:
            raise ValueError(
                "Model returned no attention weights and does not accept "
                "output_attentions=True. For HF models, load with "
                'attn_implementation="eager".') from e
        finally:
            model.train(was_training)
        attns = getattr(out, "attentions", None)

    if not attns:
        raise ValueError(
            "Model returned no attention weights. For HF models, load with "
            'attn_implementation="eager" — SDPA and flash attention do not '
            "expose them.")
    a = attns[layer_idx]
    return a[0].detach().float().cpu().numpy()      # first example: [H, Q, K]


def extract_feature_maps(model: nn.Module, image, layer_idx: int = -1) -> np.ndarray:
    """Per-head activations for one image, captured where heads are separable.

    Hooks the same attention output projection that SAL masks and FI measures —
    the projection's *input* is the concatenation of per-head attention outputs,
    so feature dimensions map cleanly onto heads there. Anywhere downstream the
    heads are already mixed.

    Returns ``[num_heads, tokens]``: each head's activation magnitude per token,
    which is what reshapes into a spatial grid for a ViT.
    """
    projs = arch_support.get_output_projections(model)
    if not projs:
        raise ValueError(
            "No attention output projections found — cannot separate heads. "
            "sal.arch_support.get_attention_modules() did not recognize this "
            "architecture.")
    proj = projs[layer_idx]

    captured = {}

    def hook(_mod, inputs):
        captured["x"] = inputs[0].detach()

    handle = proj.register_forward_pre_hook(hook)
    try:
        _run(model, image)
    finally:
        handle.remove()

    if "x" not in captured:
        raise ValueError(
            f"Layer {layer_idx}'s output projection never ran during the forward "
            "pass — the model may route around it.")

    x = captured["x"][0]                              # [tokens, hidden]
    num_heads = _num_heads(model, x.shape[-1])
    head_dim = x.shape[-1] // num_heads
    per_head = x.view(x.shape[0], num_heads, head_dim)
    return per_head.norm(dim=-1).T.float().cpu().numpy()   # [heads, tokens]


def _num_heads(model, hidden: int) -> int:
    try:
        from sal.fi import _infer_num_heads
        return _infer_num_heads(model)
    except Exception:
        logger.warning("Could not infer head count; treating the projection "
                       "input as a single head.")
        return 1


def _grid(vec: np.ndarray):
    """Reshape a token vector into the squarest possible 2-D grid.

    ViT token counts are a square plus prefix tokens (CLS, registers). Those
    prefixes are dropped rather than padded — padding invents a patch that the
    image never had. At most ``_MAX_PREFIX_TOKENS`` are dropped; beyond that the
    sequence is treated as non-spatial, because a square found by discarding a
    third of the tokens is a coincidence, not a patch grid.
    """
    n = vec.shape[0]
    side = int(np.sqrt(n))
    if side * side == n:
        return vec.reshape(side, side)
    # Drop leading prefix tokens until what remains is square.
    for drop in range(1, min(n, _MAX_PREFIX_TOKENS + 1)):
        m = n - drop
        side = int(np.sqrt(m))
        if side * side == m:
            return vec[drop:].reshape(side, side)
    return None                                        # not a spatial model


# ------------------------------------------------------------------- plotting
def visualize_attention_maps(model: nn.Module, image, layer_idx: int = -1,
                             max_heads: int = 8, save_path: Optional[str] = None,
                             title: Optional[str] = None):
    """Plot per-head attention maps for one image at one layer."""
    plt = _require_plt()
    attn = extract_attention_maps(model, image, layer_idx)
    n = min(max_heads, attn.shape[0])
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(attn[i], cmap="viridis", aspect="auto")
        ax.set_title(f"head {i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title or f"Attention maps — layer {layer_idx}")
    fig.tight_layout()
    return _finish(fig, save_path)


def visualize_feature_maps(model: nn.Module, image, layer_idx: int = -1,
                           num_features: int = 16, save_path: Optional[str] = None,
                           title: Optional[str] = None):
    """Plot per-head feature activations for one image at one layer.

    For a ViT the token axis is reshaped back into the patch grid, so each panel
    is a spatial map. For a model whose tokens are not a square grid (a language
    model, say) the panels are line plots over the sequence instead.
    """
    plt = _require_plt()
    feats = extract_feature_maps(model, image, layer_idx)
    n = min(num_features, feats.shape[0])
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        _draw_head(ax, feats[i], i)
    fig.suptitle(title or f"Feature maps — layer {layer_idx}")
    fig.tight_layout()
    return _finish(fig, save_path)


def _draw_head(ax, vec: np.ndarray, idx: int, vmin=None, vmax=None):
    grid = _grid(vec)
    if grid is not None:
        ax.imshow(grid, cmap="magma", vmin=vmin, vmax=vmax)
    else:
        ax.plot(vec, color=_SAL_COLOR, linewidth=1)
        ax.set_ylim(vmin, vmax)
    ax.set_title(f"head {idx}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def compare_feature_maps(model_original: nn.Module, model_compressed: nn.Module,
                         image, layer_idx: int = -1, num_features: int = 8,
                         labels=("original", "compressed"),
                         save_path: Optional[str] = None):
    """Side-by-side feature maps: original on top, compressed underneath.

    Both rows share a colour scale. Without that, a compressed model whose
    activations collapsed toward zero gets renormalized back to a full-range
    picture and looks *identical* to the original — the exact failure the figure
    exists to reveal.
    """
    plt = _require_plt()
    fa = extract_feature_maps(model_original, image, layer_idx)
    fb = extract_feature_maps(model_compressed, image, layer_idx)
    n = min(num_features, fa.shape[0], fb.shape[0])
    vmin = float(min(fa[:n].min(), fb[:n].min()))
    vmax = float(max(fa[:n].max(), fb[:n].max()))

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.2), squeeze=False)
    for col in range(n):
        _draw_head(axes[0][col], fa[col], col, vmin, vmax)
        _draw_head(axes[1][col], fb[col], col, vmin, vmax)
    for row, label in zip(axes, labels):
        row[0].set_ylabel(label, fontsize=10)
    fig.suptitle(f"Feature maps: {labels[0]} vs {labels[1]} (layer {layer_idx}, shared scale)")
    fig.tight_layout()
    return _finish(fig, save_path)


def visualize_compression_impact(results_dict: dict, metrics=None,
                                 save_path: Optional[str] = None,
                                 title: str = "SAL vs standard pruning"):
    """Grouped bars comparing arms of a compression benchmark.

    ``results_dict`` maps an arm name to its metric dict, as produced by
    :func:`sal.evaluation.compression_report` — for example
    ``{"sal_33": {...}, "standard_33": {...}}``. Arms whose name contains "sal"
    are coloured as the SAL arm.

    Only numeric metrics present in *every* arm are plotted; a metric missing
    from one arm would otherwise draw a bar of height zero, which reads as a
    measured zero rather than a gap in the data. Metrics are plotted on separate
    subplots because accuracy (0-1) and latency (ms) do not share an axis.
    """
    plt = _require_plt()
    if not results_dict:
        raise ValueError("results_dict is empty — nothing to plot.")

    arms = list(results_dict)
    if metrics is None:
        metrics = [m for m in results_dict[arms[0]]
                   if all(isinstance(results_dict[a].get(m), (int, float))
                          and not isinstance(results_dict[a].get(m), bool)
                          for a in arms)]
    if not metrics:
        raise ValueError(
            "No numeric metric is present in every arm. Pass metrics=[...] "
            "explicitly, or check for None entries (see the report's 'skipped').")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    x = np.arange(len(arms))
    colors = [_SAL_COLOR if "sal" in a.lower() else _BASELINE_COLOR for a in arms]
    for ax, metric in zip(axes[0], metrics):
        vals = [float(results_dict[a][metric]) for a in arms]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path)


def _finish(fig, save_path: Optional[str]):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
