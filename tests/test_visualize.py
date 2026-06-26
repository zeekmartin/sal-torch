"""Smoke tests for visual reports — PDFs generate and are non-empty (CPU)."""
import numpy as np
import pytest
import torch

from sal.compare import compare
from sal.plasticity import PlasticityScanner
from sal.scanner import FIScanner
from sal.visualize import _per_head_fragility


def _labeled_data(n_batches=5, bs=4, sl=16):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(0, 100, (bs, sl)),
             "labels": torch.randint(0, 100, (bs, sl))} for _ in range(n_batches)]


def test_per_head_fragility_shape():
    rng = np.random.RandomState(0)
    n = 32  # 4 layers x 8 heads
    adj = (rng.rand(n, n) > 0.5).astype(np.int8)
    grid = _per_head_fragility(adj, 4, 8)
    assert grid.shape == (4, 8)
    finite = grid[~np.isnan(grid)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()


def test_fi_pdf(tiny_model, probe_data, tmp_path):
    res = FIScanner(tiny_model, probe_data, num_samples=40).scan()
    out = tmp_path / "fi.pdf"
    res.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plasticity_pdf(tiny_model, probe_data, tmp_path):
    pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
    out = tmp_path / "plasticity.pdf"
    pmap.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_comparison_pdf(tiny_model, tmp_path):
    data = _labeled_data()
    res = compare(tiny_model, data, data, methods=["magnitude", "random_posthoc"], metric="loss")
    out = tmp_path / "comparison.pdf"
    res.save(str(out))
    assert out.exists() and out.stat().st_size > 0
