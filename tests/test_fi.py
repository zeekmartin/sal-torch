"""Tests for FI computation (triangle-support fragility)."""
import numpy as np
import pytest
from sal.fi import LayerClass, classify_layers, compute_fi, extract_activation_graph


class TestComputeFI:
    def test_complete_graph_robust(self):
        # Every edge has many common neighbours → fully triangulated → FI = 0.
        n = 16
        adj = np.ones((n, n), dtype=np.int8)
        np.fill_diagonal(adj, 0)
        assert compute_fi(adj) == 0.0

    def test_single_triangle_robust(self):
        tri = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
        assert compute_fi(tri) == 0.0

    def test_cycle_all_fragile(self):
        # A ring has edges but no triangles → every edge is fragile → FI = 1.
        n = 16
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
        assert compute_fi(adj) == 1.0

    def test_star_all_fragile(self):
        n = 10
        adj = np.zeros((n, n), dtype=np.int8)
        adj[0, 1:] = 1
        adj[1:, 0] = 1
        assert compute_fi(adj) == 1.0

    def test_no_edges(self):
        assert compute_fi(np.zeros((8, 8), dtype=np.int8)) == 1.0

    def test_partial_fragility(self):
        # A triangle (robust) plus a dangling edge (fragile) → FI strictly in (0, 1).
        adj = np.zeros((5, 5), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[1, 2] = adj[2, 1] = 1
        adj[0, 2] = adj[2, 0] = 1     # triangle 0-1-2
        adj[3, 4] = adj[4, 3] = 1     # isolated fragile edge
        fi = compute_fi(adj)
        assert 0.0 < fi < 1.0

    def test_range(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            n = rng.randint(4, 32)
            a = (rng.rand(n, n) > 0.5).astype(np.int8)
            a = np.triu(a, 1)
            a = a + a.T
            fi = compute_fi(a)
            assert 0.0 <= fi <= 1.0


class TestExtractGraph:
    def test_extraction_shape(self, tiny_model, probe_data):
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        assert adj.shape == (32, 32)

    def test_extraction_binary(self, tiny_model, probe_data):
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        assert set(np.unique(adj)).issubset({0, 1})
        assert np.all(np.diag(adj) == 0)
        assert np.array_equal(adj, adj.T)  # undirected


class TestClassify:
    def test_types(self, tiny_model, probe_data):
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        lm = classify_layers(tiny_model, adj)
        assert len(lm) == 4
        for c in lm.values():
            assert c in (LayerClass.IMMUNE, LayerClass.BUFFER, LayerClass.CRITICAL)
