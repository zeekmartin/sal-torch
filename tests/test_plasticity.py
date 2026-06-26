"""Tests for PlasticityScanner (CPU, tiny model fixture)."""
import json

import numpy as np
import pytest
import torch

from sal.plasticity import (ELASTIC, HUB, SATURATED, PlasticityMap, PlasticityScanner,
                            Recommendation, _attention_entropy, _linear_cka,
                            _mean_abs_offdiag_corr)


class TestAxisHelpers:
    def test_routing_entropy_uniform_is_max(self):
        # Uniform attention over S keys -> normalized entropy == 1.
        attn = torch.ones(2, 8, 5, 5) / 5
        ent = _attention_entropy(attn)
        assert ent.shape == (8,)
        assert np.allclose(ent, 1.0, atol=1e-4)

    def test_routing_entropy_peaked_is_min(self):
        # One-hot attention -> zero entropy.
        attn = torch.zeros(2, 4, 6, 6)
        attn[..., 0] = 1.0
        ent = _attention_entropy(attn)
        assert np.allclose(ent, 0.0, atol=1e-4)

    def test_cka_identical_is_one(self):
        rng = np.random.RandomState(0)
        x = rng.randn(20, 16)
        assert _linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)
        assert _linear_cka(x, 3.0 * x) == pytest.approx(1.0, abs=1e-6)  # scale invariant

    def test_cka_range(self):
        rng = np.random.RandomState(1)
        for _ in range(10):
            x = rng.randn(30, 12); y = rng.randn(30, 12)
            assert 0.0 <= _linear_cka(x, y) <= 1.0

    def test_mi_identical_heads_is_one(self):
        row = np.random.RandomState(2).randn(40)
        mat = np.stack([row, row, row])
        assert _mean_abs_offdiag_corr(mat) == pytest.approx(1.0, abs=1e-6)

    def test_mi_range(self):
        rng = np.random.RandomState(3)
        mat = rng.randn(8, 50)
        assert 0.0 <= _mean_abs_offdiag_corr(mat) <= 1.0


class TestScan:
    def test_scan_structure(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        assert isinstance(pmap, PlasticityMap)
        assert pmap.num_layers == 4
        assert pmap.num_heads_per_layer == 8

    def test_routing_axis(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        assert set(pmap.routing.keys()) == {0, 1, 2, 3}
        for v in pmap.routing.values():
            assert 0.0 <= v <= 1.0

    def test_cka_axis(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        assert set(pmap.cka_similarity.keys()) == {(0, 1), (1, 2), (2, 3)}
        for v in pmap.cka_similarity.values():
            assert 0.0 <= v <= 1.0

    def test_mi_axis(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        assert set(pmap.mutual_info.keys()) == {0, 1, 2, 3}
        for v in pmap.mutual_info.values():
            assert 0.0 <= v <= 1.0

    def test_absorption_map(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        assert set(pmap.absorption_map.keys()) == {0, 1, 2, 3}
        for v in pmap.absorption_map.values():
            assert v in (ELASTIC, SATURATED, HUB)

    def test_summary_and_props(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        assert isinstance(pmap.summary, str)
        assert len(pmap.elastic_layers) + len(pmap.saturated_layers) + len(pmap.hub_layers) == 4


class TestRecommend:
    def test_recommend_format(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        rec = pmap.recommend(target_compression=0.33)
        assert isinstance(rec, Recommendation)
        assert rec.target_compression == 0.33
        assert isinstance(rec.safe_to_prune, list)
        assert isinstance(rec.never_touch, list)
        assert isinstance(rec.expected_impact, float)
        for entry in rec.safe_to_prune:
            assert len(entry) == 2  # (layer, head)
        # never_touch are heads in hub layers
        for (layer, head) in rec.never_touch:
            assert layer in pmap.hub_layers

    def test_recommend_count(self, tiny_model, probe_data):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        rec = pmap.recommend(target_compression=0.33)
        # at most the target number of heads, never overlapping never_touch
        assert len(rec.safe_to_prune) <= round(0.33 * 32)
        assert not (set(rec.safe_to_prune) & set(rec.never_touch))


class TestSave:
    def test_save_json(self, tiny_model, probe_data, tmp_path):
        pmap = PlasticityScanner(tiny_model, probe_data, num_samples=40).scan()
        out = tmp_path / "plasticity.json"
        pmap.save(str(out))
        data = json.loads(out.read_text())
        assert data["num_layers"] == 4
        assert "absorption_map" in data and "routing" in data
