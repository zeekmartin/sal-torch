"""Feature-map and attention-map visualization (v0.5.1).

CPU-only, no checkpoints: the tiny transformer stands in for a ViT, and a
tiny conv-attention model stands in for the image path. These are smoke tests
with teeth — they check the extracted arrays have the right *shape* and that
the comparison figure really shares a colour scale, because a figure that
renders is not the same as a figure that is correct.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("matplotlib")

from sal.visualization import (                                     # noqa: E402
    compare_feature_maps, extract_attention_maps, extract_feature_maps,
    visualize_attention_maps, visualize_compression_impact, visualize_feature_maps,
)
from sal.visualization import _grid                                 # noqa: E402


@pytest.fixture
def image_batch():
    """A dict batch — the tiny model is a token model, so tokens stand in."""
    torch.manual_seed(0)
    return {"input_ids": torch.randint(0, 100, (1, 16))}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# --------------------------------------------------------------- grid reshaping
def test_grid_square_token_count():
    assert _grid(np.arange(16.0)).shape == (4, 4)


def test_grid_drops_prefix_tokens():
    """197 ViT tokens = 196 patches + CLS. The CLS token is dropped, not padded."""
    g = _grid(np.arange(197.0))
    assert g.shape == (14, 14)
    assert g[0, 0] == 1.0          # token 0 (CLS) was dropped


def test_grid_returns_none_for_non_spatial():
    """47 tokens: no square within the prefix budget (36 and 49 are too far)."""
    assert _grid(np.arange(47.0)) is None


def test_grid_refuses_implausible_prefix():
    """23 tokens would be square after dropping 7 — that is a coincidence."""
    assert _grid(np.arange(23.0)) is None


# ------------------------------------------------------------------ extraction
def test_feature_map_extraction(tiny_model, image_batch):
    """Shape is [heads, tokens] — heads separable at the projection input."""
    feats = extract_feature_maps(tiny_model, image_batch)
    assert feats.shape == (8, 16)          # 8 heads, 16 tokens
    assert np.isfinite(feats).all()
    assert (feats >= 0).all()              # per-head norms


def test_feature_map_extraction_per_layer(tiny_model, image_batch):
    """Different layers give different activations — the index is honoured."""
    first = extract_feature_maps(tiny_model, image_batch, layer_idx=0)
    last = extract_feature_maps(tiny_model, image_batch, layer_idx=-1)
    assert first.shape == last.shape
    assert not np.allclose(first, last)


def test_feature_map_extraction_removes_its_hook(tiny_model, image_batch):
    """The capture hook must not survive the call."""
    from sal import arch_support
    proj = arch_support.get_output_projections(tiny_model)[-1]
    before = len(proj._forward_pre_hooks)
    extract_feature_maps(tiny_model, image_batch)
    assert len(proj._forward_pre_hooks) == before


def test_feature_map_extraction_restores_training_mode(tiny_model, image_batch):
    tiny_model.train()
    extract_feature_maps(tiny_model, image_batch)
    assert tiny_model.training is True


def test_feature_map_extraction_unknown_architecture(image_batch):
    model = nn.Linear(4, 4)                 # no attention anywhere
    with pytest.raises(ValueError, match="No attention output projections"):
        extract_feature_maps(model, torch.randn(3, 8, 8))


def test_attention_map_extraction(tiny_model, image_batch):
    """Shape is [heads, queries, keys] and each row is a distribution."""
    attn = extract_attention_maps(tiny_model, image_batch)
    assert attn.shape == (8, 16, 16)
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-4)


def test_attention_map_extraction_without_support():
    """A model with no attentions gets a message that says what to do."""
    class NoAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x)

    with pytest.raises(ValueError, match="attn_implementation"):
        extract_attention_maps(NoAttn(), torch.randn(1, 3, 4, 4))


# -------------------------------------------------------------------- plotting
def test_visualize_feature_maps_returns_figure(tiny_model, image_batch):
    import matplotlib.figure
    fig = visualize_feature_maps(tiny_model, image_batch, num_features=4)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 4


def test_visualize_attention_maps_returns_figure(tiny_model, image_batch):
    import matplotlib.figure
    fig = visualize_attention_maps(tiny_model, image_batch, max_heads=4)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 4


def test_comparison_plot_generates_figure(tiny_model, image_batch):
    """Two rows of panels: original on top, compressed underneath."""
    import copy
    import matplotlib.figure
    compressed = copy.deepcopy(tiny_model)
    with torch.no_grad():                               # crude "compression"
        compressed.transformer.h[-1].attn.out_proj.weight.mul_(0.1)

    fig = compare_feature_maps(tiny_model, compressed, image_batch, num_features=4)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 8                           # 2 rows x 4 heads


def test_comparison_plot_shares_colour_scale(tiny_model, image_batch):
    """A collapsed model must *look* collapsed, not be renormalized back."""
    import copy
    compressed = copy.deepcopy(tiny_model)
    with torch.no_grad():
        for blk in compressed.transformer.h:
            blk.attn.v_proj.weight.mul_(0.01)

    fig = compare_feature_maps(tiny_model, compressed, image_batch, num_features=3)
    clims = [im.get_clim() for ax in fig.axes for im in ax.get_images()]
    assert clims, "no image panels were drawn"
    assert len(set(clims)) == 1, "panels were independently normalized"


def test_save_to_file(tiny_model, image_batch, tmp_path):
    out = tmp_path / "features.png"
    visualize_feature_maps(tiny_model, image_batch, num_features=4, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_compare_save_to_file(tiny_model, image_batch, tmp_path):
    import copy
    out = tmp_path / "compare.png"
    compare_feature_maps(tiny_model, copy.deepcopy(tiny_model), image_batch,
                         num_features=2, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------------- compression impact bars
def _results():
    return {
        "sal_33":      {"knn_accuracy": 0.71, "cka_similarity": 0.94, "latency_cpu_ms": 12.0},
        "standard_33": {"knn_accuracy": 0.63, "cka_similarity": 0.88, "latency_cpu_ms": 12.1},
    }


def test_visualize_compression_impact(tmp_path):
    import matplotlib.figure
    out = tmp_path / "impact.png"
    fig = visualize_compression_impact(_results(), save_path=str(out))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3                       # one subplot per metric
    assert out.exists() and out.stat().st_size > 0


def test_visualize_compression_impact_selects_metrics():
    fig = visualize_compression_impact(_results(), metrics=["knn_accuracy"])
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "knn_accuracy"


def test_visualize_compression_impact_colours_sal_arm():
    fig = visualize_compression_impact(_results(), metrics=["knn_accuracy"])
    colors = [p.get_facecolor() for p in fig.axes[0].patches]
    assert len(set(colors)) == 2, "SAL and baseline arms should differ in colour"


def test_visualize_compression_impact_skips_partial_metrics():
    """A metric missing from one arm is dropped, not drawn as a zero."""
    res = _results()
    res["standard_33"]["knn_accuracy"] = None       # e.g. probe was skipped
    fig = visualize_compression_impact(res)
    titles = [ax.get_title() for ax in fig.axes]
    assert "knn_accuracy" not in titles
    assert "cka_similarity" in titles


def test_visualize_compression_impact_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        visualize_compression_impact({})


def test_visualize_compression_impact_no_numeric_metric_raises():
    with pytest.raises(ValueError, match="No numeric metric"):
        visualize_compression_impact({"a": {"note": "x"}, "b": {"note": "y"}})
