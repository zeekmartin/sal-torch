"""Visual reports — heatmaps and one-page PDF summaries.

Turns the numeric outputs of the scanners into something a human can read at a
glance: an FI fragility heatmap, the three-axis plasticity bars, a colour-coded
absorption map, and a method-comparison bar chart. Everything is optional —
``pip install sal-torch[reports]`` pulls in matplotlib + fpdf2.

The public entry points are the ``render_*_pdf`` functions, which the result
objects call from their ``.save("report.pdf")`` methods. The per-axes drawing
helpers are exposed too, so callers can compose their own figures.
"""
from __future__ import annotations

import io

import numpy as np

_ELASTIC_COLOR = "#2e7d32"   # green  — safe to compress
_SATURATED_COLOR = "#c62828"  # red    — structural bottleneck
_HUB_COLOR = "#ef6c00"        # orange — compensates for others
_CLASS_COLORS = {"ELASTIC": _ELASTIC_COLOR, "SATURATED": _SATURATED_COLOR, "HUB": _HUB_COLOR}

# FI layer classes share the colour language (immune≈elastic, critical≈saturated).
_FI_CLASS_COLORS = {"IMMUNE": _ELASTIC_COLOR, "BUFFER": _HUB_COLOR, "CRITICAL": _SATURATED_COLOR}


def _require():
    """Import the optional plotting stack with a friendly error."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display needed
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without extras
        raise ImportError("Visual reports need matplotlib + fpdf2. "
                          "Install with: pip install sal-torch[reports]") from e
    try:
        from fpdf import FPDF
    except ImportError as e:  # pragma: no cover
        raise ImportError("Visual reports need fpdf2. "
                          "Install with: pip install sal-torch[reports]") from e
    return plt, FPDF


def _fig_to_png(fig) -> io.BytesIO:
    """Render a matplotlib figure to an in-memory PNG and close it."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------------------------------------------------------- per-head FI grid
def _per_head_fragility(adjacency: np.ndarray, nl: int, nh: int) -> np.ndarray:
    """Per-node fragility reshaped to (layers, heads).

    A head's fragility is the fraction of its edges that have *no* triangle
    support — the same zero-triangle signal the scalar FI aggregates, kept
    per-node so the heatmap shows *where* the fragility sits.
    """
    A = (np.asarray(adjacency) != 0).astype(np.int64)
    np.fill_diagonal(A, 0)
    AA = A @ A  # AA[i, j] = number of common neighbours = triangle support of edge (i, j)
    n = A.shape[0]
    frag = np.zeros(n)
    for i in range(n):
        nbrs = A[i] > 0
        deg = int(nbrs.sum())
        if deg == 0:
            frag[i] = 0.0
        else:
            frag[i] = float((AA[i][nbrs] == 0).sum()) / deg
    grid = np.full((nl, nh), np.nan)
    for idx in range(min(n, nl * nh)):
        grid[idx // nh, idx % nh] = frag[idx]
    return grid


# --------------------------------------------------------------------- drawing API
def draw_fi_heatmap(scan_result, ax):
    """Layer×head fragility heatmap onto ``ax``."""
    grid = _per_head_fragility(scan_result.adjacency, scan_result.num_layers,
                               scan_result.num_heads_per_layer)
    im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Fragility per head  (FI={scan_result.fi_score:.3f})")
    ax.set_yticks(range(scan_result.num_layers))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="zero-triangle fraction")
    return ax


def _cka_per_layer(pmap) -> dict:
    cka_layer = {}
    for li in range(pmap.num_layers):
        vals = [v for k, v in pmap.cka_similarity.items() if li in k]
        cka_layer[li] = max(vals) if vals else 0.0
    return cka_layer


def draw_plasticity_bars(pmap, ax):
    """Grouped three-axis bars (routing / CKA / MI) per layer onto ``ax``."""
    layers = list(range(pmap.num_layers))
    cka_layer = _cka_per_layer(pmap)
    routing = [0.0 if np.isnan(pmap.routing.get(l, np.nan)) else pmap.routing[l] for l in layers]
    cka = [cka_layer[l] for l in layers]
    mi = [pmap.mutual_info.get(l, 0.0) for l in layers]

    x = np.arange(len(layers))
    w = 0.27
    ax.bar(x - w, routing, w, label="routing", color="#1565c0")
    ax.bar(x, cka, w, label="CKA (neighbour)", color="#6a1b9a")
    ax.bar(x + w, mi, w, label="MI (heads)", color="#00838f")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("layer")
    ax.set_ylabel("score [0-1]")
    ax.set_ylim(0, 1.05)
    ax.set_title("Plasticity axes")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def draw_absorption_map(pmap, ax):
    """Colour-coded per-layer absorption strip onto ``ax``."""
    layers = list(range(pmap.num_layers))
    colors = [_CLASS_COLORS.get(pmap.absorption_map.get(l, ""), "#9e9e9e") for l in layers]
    ax.bar(range(len(layers)), [1] * len(layers), color=colors, width=0.9)
    for i, l in enumerate(layers):
        label = pmap.absorption_map.get(l, "?")
        ax.text(i, 0.5, label, rotation=90, ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticks([])
    ax.set_xlabel("layer")
    ax.set_title("Absorption map")
    return ax


def draw_guard_map(guard, ax):
    """Protected (red) vs trainable (green) heads per layer, onto ``ax``."""
    nl, nh = guard.num_layers, guard.num_heads
    grid = np.zeros((nl, nh))
    for (l, h) in guard.protected_heads:
        if 0 <= l < nl and 0 <= h < nh:
            grid[l, h] = 1.0
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([_ELASTIC_COLOR, _SATURATED_COLOR])  # 0 trainable, 1 protected
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_yticks(range(nl))
    n_prot = len(guard.protected_heads)
    ax.set_title(f"Structural guard  ({n_prot}/{nl * nh} heads protected, "
                 f"level={guard.protection_level:.2f})")
    return ax


def draw_drift_bars(report, ax):
    """Per-layer retention bars (CKA, 1=identical); reclassified layers in red."""
    layers = sorted(report.layer_drift.keys())
    vals = [0.0 if np.isnan(report.layer_drift[l]) else report.layer_drift[l] for l in layers]
    colors = [_SATURATED_COLOR if l in report.classification_changes else _ELASTIC_COLOR
              for l in layers]
    bars = ax.bar(range(len(layers)), vals, color=colors, width=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("layer")
    ax.set_ylabel("activation retention (CKA)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Structural drift  (forgetting={report.forgetting_score:.3f})")
    return ax


def draw_comparison_bars(compare_result, ax):
    """Method-vs-score bars, winner highlighted, onto ``ax``."""
    results = compare_result.results
    names = [r.method for r in results]
    scores = [r.score for r in results]
    winner = compare_result.winner
    colors = [_ELASTIC_COLOR if r.method == winner else "#90a4ae" for r in results]
    bars = ax.bar(range(len(names)), scores, color=colors, width=0.6)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{s:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(compare_result.metric)
    ax.set_title(f"Method comparison  (winner: {winner})")
    return ax


# --------------------------------------------------------------------- PDF writers
def _ascii(text: str) -> str:
    """Down-map to latin-1 so the built-in PDF fonts can render it."""
    repl = {"—": "-", "–": "-", "→": "->", "±": "+/-",
            "≈": "~", "≥": ">=", "≤": "<="}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _pdf_with_image(FPDF, title: str, lines, png: io.BytesIO, path: str):
    """Assemble a one-page PDF: title, summary lines, then the figure."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _ascii(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in lines:
        pdf.multi_cell(0, 6, _ascii(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.image(png, w=pdf.epw)  # full content width
    pdf.output(path)


def render_fi_pdf(scan_result, path: str):
    """One-page FI report: summary + per-head fragility heatmap."""
    plt, FPDF = _require()
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_fi_heatmap(scan_result, ax)
    png = _fig_to_png(fig)
    lines = [
        scan_result.summary,
        f"layers={scan_result.num_layers}, heads/layer={scan_result.num_heads_per_layer}",
        "Brighter cells are more fragile (more edges without triangle support).",
    ]
    _pdf_with_image(FPDF, "SAL — Fragility Report", lines, png, path)


def render_plasticity_pdf(pmap, path: str):
    """One-page plasticity report: axes bars + absorption strip + recommendation."""
    plt, FPDF = _require()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), height_ratios=[3, 1])
    draw_plasticity_bars(pmap, ax1)
    draw_absorption_map(pmap, ax2)
    fig.tight_layout()
    png = _fig_to_png(fig)

    rec = pmap.recommend()
    lines = [
        pmap.summary,
        f"layers={pmap.num_layers}, heads/layer={pmap.num_heads_per_layer}",
        f"recommend (33% target): prune {len(rec.safe_to_prune)} heads, "
        f"never-touch {len(rec.never_touch)} heads, est. impact {rec.expected_impact:+.3f}",
    ]
    _pdf_with_image(FPDF, "SAL — Plasticity Report", lines, png, path)


def render_guard_pdf(guard, path: str):
    """One-page guard report: protected/trainable head map + protection summary."""
    plt, FPDF = _require()
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_guard_map(guard, ax)
    fig.tight_layout()
    png = _fig_to_png(fig)
    pm = guard.protection_map
    lines = [
        f"protection level={guard.protection_level:.2f}  |  "
        f"{len(guard.protected_heads)} protected, {len(guard.trainable_heads)} trainable",
        f"layers={guard.num_layers}, heads/layer={guard.num_heads}, "
        f"protected layers touched={len(pm)}",
        "Red = protected (frozen via gradient masking). Green = trainable (absorbs new task).",
    ]
    _pdf_with_image(FPDF, "SAL - Structural Guard", lines, png, path)


def render_drift_pdf(report, path: str):
    """One-page drift report: per-layer retention bars + forgetting summary."""
    plt, FPDF = _require()
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_drift_bars(report, ax)
    fig.tight_layout()
    png = _fig_to_png(fig)
    changes = ", ".join(f"L{l}:{a}->{b}" for l, (a, b) in report.classification_changes.items())
    lines = [
        report.summary,
        f"FI before={report.fi_before:.3f}, after={report.fi_after:.3f}",
        f"reclassified layers: {changes or 'none'}",
        "Taller bars = more of the layer's representation was retained.",
    ]
    _pdf_with_image(FPDF, "SAL - Structural Drift", lines, png, path)


def render_comparison_pdf(compare_result, path: str):
    """One-page comparison report: bar chart + the text table."""
    plt, FPDF = _require()
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_comparison_bars(compare_result, ax)
    fig.tight_layout()
    png = _fig_to_png(fig)
    lines = [
        f"metric={compare_result.metric}, compression={compare_result.compression}",
        f"winner: {compare_result.winner}",
    ] + compare_result.table.splitlines()
    _pdf_with_image(FPDF, "SAL — Method Comparison", lines, png, path)
