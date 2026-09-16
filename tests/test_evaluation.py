"""Representation-quality metrics (v0.5.1).

All CPU, all synthetic — no ImageNet, no checkpoints, no GPU. The models here
are deliberately trivial so the *expected value* of each metric is known in
advance: a probe on linearly separable features should be near-perfect, CKA
against an identical model should be exactly 1.
"""
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sal.evaluation import (
    cka_similarity, compression_report, count_params, extract_features,
    knn_accuracy, linear_probe, measure_latency, representation_similarity,
)


class TinyEncoder(nn.Module):
    """Flat [B, D_in] -> [B, D_out] encoder. Stands in for a ViT backbone."""
    def __init__(self, d_in=16, d_out=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x)


class TokenEncoder(nn.Module):
    """Returns [B, tokens, D] — the shape a ViT actually emits, to test pooling."""
    def __init__(self, d_in=16, d_out=8, tokens=4):
        super().__init__()
        self.tokens = tokens
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x).unsqueeze(1).expand(-1, self.tokens, -1)


class ConvEncoder(nn.Module):
    """Returns [B, C, H, W] — tests the global-average-pool branch."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)


def _separable(n=120, d=16, num_classes=3, seed=0):
    """Well-separated Gaussian clusters — a linear probe should nail these."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % num_classes
    centers = torch.zeros(num_classes, d)
    for c in range(num_classes):
        centers[c, c] = 10.0
    x = centers[y] + 0.1 * torch.randn(n, d, generator=g)
    return x, y


@pytest.fixture
def labeled_loaders():
    xtr, ytr = _separable(120, seed=0)
    xva, yva = _separable(60, seed=1)
    return (DataLoader(TensorDataset(xtr, ytr), batch_size=16),
            DataLoader(TensorDataset(xva, yva), batch_size=16))


@pytest.fixture
def plain_loader():
    g = torch.Generator().manual_seed(7)
    return DataLoader(TensorDataset(torch.randn(48, 16, generator=g)), batch_size=8)


# ------------------------------------------------------------ feature extraction
def test_extract_features_shape_and_labels(labeled_loaders):
    train_loader, _ = labeled_loaders
    feats, labels = extract_features(TinyEncoder(), train_loader)
    assert feats.shape == (120, 8)
    assert feats.dtype == torch.float32
    assert labels is not None and labels.shape == (120,)


def test_extract_features_no_labels(plain_loader):
    feats, labels = extract_features(TinyEncoder(), plain_loader)
    assert feats.shape == (48, 8)
    assert labels is None


def test_extract_features_pools_token_axis(labeled_loaders):
    """[B, tokens, D] output is mean-pooled down to [B, D]."""
    train_loader, _ = labeled_loaders
    feats, _ = extract_features(TokenEncoder(), train_loader)
    assert feats.shape == (120, 8)


def test_extract_features_pools_spatial_axes():
    """[B, C, H, W] output is global-average-pooled down to [B, C]."""
    g = torch.Generator().manual_seed(0)
    dl = DataLoader(TensorDataset(torch.randn(8, 3, 8, 8, generator=g)), batch_size=4)
    feats, _ = extract_features(ConvEncoder(), dl)
    assert feats.shape == (8, 6)


def test_extract_features_respects_num_batches(labeled_loaders):
    train_loader, _ = labeled_loaders
    feats, _ = extract_features(TinyEncoder(), train_loader, num_batches=2)
    assert feats.shape[0] == 32          # 2 batches of 16


def test_extract_features_restores_training_mode(labeled_loaders):
    train_loader, _ = labeled_loaders
    model = TinyEncoder().train()
    extract_features(model, train_loader)
    assert model.training is True


def test_extract_features_dict_batches(tiny_model, probe_data):
    """Dict batches are forwarded as kwargs — the HF/tiny-model convention."""
    feats, labels = extract_features(tiny_model, probe_data)
    assert feats.shape[0] == 4 * len(probe_data)
    assert labels is None


def test_extract_features_custom_feature_fn(labeled_loaders):
    train_loader, _ = labeled_loaders
    out = extract_features(TinyEncoder(), train_loader,
                           feature_fn=lambda m, x: torch.ones(x.shape[0], 5))
    assert out[0].shape == (120, 5)
    assert torch.all(out[0] == 1.0)


# ------------------------------------------------------------------ linear probe
def test_linear_probe_returns_accuracy(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    acc = linear_probe(nn.Identity(), train_loader, val_loader, epochs=20)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_separable_data_is_accurate(labeled_loaders):
    """Well-separated clusters: the probe should be essentially perfect."""
    train_loader, val_loader = labeled_loaders
    assert linear_probe(nn.Identity(), train_loader, val_loader) > 0.95


def test_linear_probe_random_features_are_chance(labeled_loaders):
    """Features carrying no label information sit at chance (1/3 here)."""
    train_loader, val_loader = labeled_loaders
    acc = linear_probe(nn.Identity(), train_loader, val_loader,
                       feature_fn=lambda m, x: torch.randn(x.shape[0], 8))
    assert acc < 0.7


def test_linear_probe_infers_num_classes(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    explicit = linear_probe(nn.Identity(), train_loader, val_loader, num_classes=3, epochs=10)
    inferred = linear_probe(nn.Identity(), train_loader, val_loader, epochs=10)
    assert explicit == pytest.approx(inferred)


def test_linear_probe_needs_labels(plain_loader):
    with pytest.raises(ValueError, match="needs labels"):
        linear_probe(TinyEncoder(), plain_loader, plain_loader)


# --------------------------------------------------------------------- kNN
def test_knn_accuracy_basic(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    acc = knn_accuracy(nn.Identity(), train_loader, val_loader, k=5)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_knn_accuracy_separable_data_is_accurate(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    assert knn_accuracy(nn.Identity(), train_loader, val_loader, k=5) > 0.95


def test_knn_accuracy_random_features_are_chance(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    acc = knn_accuracy(nn.Identity(), train_loader, val_loader, k=5,
                       feature_fn=lambda m, x: torch.randn(x.shape[0], 8))
    assert acc < 0.7


def test_knn_k_larger_than_trainset_is_clamped(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    acc = knn_accuracy(nn.Identity(), train_loader, val_loader, k=10_000)
    assert 0.0 <= acc <= 1.0


def test_knn_needs_labels(plain_loader):
    with pytest.raises(ValueError, match="needs labels"):
        knn_accuracy(TinyEncoder(), plain_loader, plain_loader)


# --------------------------------------------------------------------- CKA
def test_cka_identical_models(plain_loader):
    """The same model against itself is CKA 1.0."""
    model = TinyEncoder()
    assert cka_similarity(model, model, plain_loader) == pytest.approx(1.0, abs=1e-6)


def test_cka_identical_weights_different_objects(plain_loader):
    import copy
    a = TinyEncoder()
    b = copy.deepcopy(a)
    assert cka_similarity(a, b, plain_loader) == pytest.approx(1.0, abs=1e-6)


def test_cka_different_models(plain_loader):
    a, b = TinyEncoder(seed=0), TinyEncoder(seed=99)
    cka = cka_similarity(a, b, plain_loader)
    assert 0.0 <= cka < 1.0


def test_cka_invariant_to_feature_scaling(plain_loader):
    """CKA ignores an invertible linear map — that is the point of using it."""
    model = TinyEncoder()
    scaled = cka_similarity(model, model, plain_loader,
                            feature_fn=None)
    assert scaled == pytest.approx(1.0, abs=1e-6)

    a_fn = lambda m, x: m(x)
    b_fn = lambda m, x: m(x) * 7.5
    fa, _ = extract_features(model, plain_loader, feature_fn=a_fn)
    fb, _ = extract_features(model, plain_loader, feature_fn=b_fn)
    from sal.plasticity import _linear_cka
    assert _linear_cka(fa.numpy().astype("float64"),
                       fb.numpy().astype("float64")) == pytest.approx(1.0, abs=1e-6)


def test_cka_handles_different_widths(plain_loader):
    """An original and a narrowed model can still be compared."""
    wide, narrow = TinyEncoder(d_out=8), TinyEncoder(d_out=4)
    cka = cka_similarity(wide, narrow, plain_loader)
    assert 0.0 <= cka <= 1.0


def test_cka_respects_num_batches(plain_loader):
    model = TinyEncoder()
    assert cka_similarity(model, model, plain_loader, num_batches=2) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------- cosine similarity
def test_representation_similarity_identical_is_one(plain_loader):
    model = TinyEncoder()
    assert representation_similarity(model, model, plain_loader) == pytest.approx(1.0, abs=1e-5)


def test_representation_similarity_different_models(plain_loader):
    sim = representation_similarity(TinyEncoder(seed=0), TinyEncoder(seed=99), plain_loader)
    assert -1.0 <= sim < 1.0


def test_representation_similarity_rejects_shape_mismatch(plain_loader):
    with pytest.raises(ValueError, match="Feature shapes differ"):
        representation_similarity(TinyEncoder(d_out=8), TinyEncoder(d_out=4), plain_loader)


# ----------------------------------------------------------------- the budget
def test_measure_latency_returns_float():
    ms = measure_latency(TinyEncoder(), input_shape=(1, 16), device="cpu",
                         warmup=2, runs=5)
    assert isinstance(ms, float)
    assert ms > 0.0 and math.isfinite(ms)


def test_measure_latency_restores_model_state():
    model = TinyEncoder().train()
    measure_latency(model, input_shape=(1, 16), device="cpu", warmup=1, runs=2)
    assert model.training is True
    assert next(model.parameters()).device.type == "cpu"


def test_measure_latency_custom_input_fn(tiny_model):
    ms = measure_latency(
        tiny_model, device="cpu", warmup=1, runs=3,
        input_fn=lambda dev: {"input_ids": torch.randint(0, 100, (1, 16), device=dev)})
    assert ms > 0.0


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present")
def test_measure_latency_cuda_unavailable_raises():
    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        measure_latency(TinyEncoder(), device="cuda")


def test_count_params():
    model = nn.Linear(4, 3)                      # 4*3 weights + 3 biases
    assert count_params(model) == 15


def test_count_params_only_trainable():
    model = nn.Linear(4, 3)
    model.weight.requires_grad_(False)
    assert count_params(model) == 15
    assert count_params(model, only_trainable=True) == 3


# ---------------------------------------------------------------- the report
def test_compression_report_core_fields(plain_loader):
    original, compressed = TinyEncoder(d_out=8), TinyEncoder(d_out=4)
    r = compression_report(original, compressed, plain_loader, name="prune50",
                           input_shape=(1, 16))

    assert r["name"] == "prune50"
    assert r["original_params"] == count_params(original)
    assert r["compressed_params"] == count_params(compressed)
    assert r["compression_ratio"] > 1.0
    assert 0.0 <= r["cka_similarity"] <= 1.0
    assert r["latency_cpu_ms"] > 0.0
    assert r["latency_cpu_ms_original"] > 0.0


def test_compression_report_reports_latency_failure(plain_loader):
    """A model that rejects the timing input loses latency, not the whole report."""
    r = compression_report(TinyEncoder(), TinyEncoder(seed=1), plain_loader)
    assert r["latency_cpu_ms"] is None
    assert "latency_cpu_ms" in r["skipped"]
    assert r["cka_similarity"] is not None      # the quality metrics survived


def test_compression_report_explains_skips(plain_loader):
    """A None in a benchmark table must come with a reason."""
    r = compression_report(TinyEncoder(), TinyEncoder(seed=1), plain_loader,
                           input_shape=(1, 16))
    assert r["linear_probe"] is None
    assert "linear_probe" in r["skipped"]
    assert "fi" in r["skipped"]


def test_compression_report_with_probes(labeled_loaders):
    train_loader, val_loader = labeled_loaders
    r = compression_report(nn.Identity(), nn.Identity(), train_loader,
                           train_loader=train_loader, val_loader=val_loader,
                           name="identity", input_shape=(1, 16))
    assert r["linear_probe"] > 0.9
    assert r["knn_accuracy"] > 0.9
    assert "linear_probe" not in r.get("skipped", {})
    # Identity against identity: nothing was compressed, nothing changed.
    assert r["cka_similarity"] == pytest.approx(1.0, abs=1e-6)
