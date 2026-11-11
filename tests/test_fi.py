"""Tests for FI computation (triangle-support fragility)."""
import numpy as np
from sal.fi import (LayerClass, classify_layers, classify_layers_by_rank,
                    compute_fi, extract_activation_graph, layer_fi_deltas)


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

    def test_default_critical_thr_is_the_measured_one(self):
        """exp30b: 0.05 saturated, 0.25 did not. The default carries that."""
        import inspect
        sig = inspect.signature(classify_layers)
        assert sig.parameters["critical_thr"].default == 0.25
        assert sig.parameters["immune_thr"].default == 0.01

    def test_thresholds_are_still_overridable(self, tiny_model, probe_data):
        """The old behaviour must stay reachable: 0.25 is one architecture."""
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        loose = classify_layers(tiny_model, adj, critical_thr=0.05)
        strict = classify_layers(tiny_model, adj, critical_thr=0.25)
        n_loose = sum(c is LayerClass.CRITICAL for c in loose.values())
        n_strict = sum(c is LayerClass.CRITICAL for c in strict.values())
        # A higher bar can never call more layers critical.
        assert n_strict <= n_loose

    def test_classify_agrees_with_deltas(self, tiny_model, probe_data):
        """classify_layers must be exactly a threshold on layer_fi_deltas."""
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        deltas = layer_fi_deltas(tiny_model, adj)
        lm = classify_layers(tiny_model, adj, immune_thr=0.01, critical_thr=0.25)
        assert set(deltas) == set(lm)
        for li, d in deltas.items():
            expected = (LayerClass.IMMUNE if d < 0.01
                        else LayerClass.CRITICAL if d >= 0.25
                        else LayerClass.BUFFER)
            assert lm[li] is expected, (li, d)


class TestLayerFIDeltas:
    def test_shape_and_sign(self, tiny_model, probe_data):
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        deltas = layer_fi_deltas(tiny_model, adj)
        assert len(deltas) == 4
        assert all(isinstance(v, float) and v >= 0.0 for v in deltas.values())

    def test_zero_fi_graph_is_all_zero(self, tiny_model):
        """A fully triangulated graph has FI 0; there is nothing to be relative to."""
        adj = np.ones((32, 32), dtype=int)
        np.fill_diagonal(adj, 0)
        deltas = layer_fi_deltas(tiny_model, adj, num_heads_per_layer=8)
        assert deltas == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    def test_deltas_are_reusable(self, tiny_model, probe_data):
        """The point of the function: any k, any threshold, no model needed."""
        adj = extract_activation_graph(tiny_model, probe_data, num_samples=20)
        deltas = layer_fi_deltas(tiny_model, adj)
        for k in range(5):
            assert len(classify_layers_by_rank(deltas, k=k)) == min(k, 4)


class TestClassifyByRank:
    def test_returns_k_layers_most_sensitive_first(self):
        deltas = {0: 0.10, 1: 0.90, 2: 0.50, 3: 0.02}
        assert classify_layers_by_rank(deltas, k=3) == [1, 2, 0]
        assert classify_layers_by_rank(deltas, k=1) == [1]

    def test_ties_break_on_lower_index(self):
        """Determinism matters: a certificate must not reorder between runs."""
        deltas = {0: 0.5, 1: 0.9, 2: 0.9, 3: 0.1}
        assert classify_layers_by_rank(deltas, k=2) == [1, 2]

    def test_k_clamped_and_non_positive(self):
        deltas = {0: 0.1, 1: 0.2}
        assert classify_layers_by_rank(deltas, k=99) == [1, 0]
        assert classify_layers_by_rank(deltas, k=0) == []
        assert classify_layers_by_rank(deltas, k=-1) == []

    def test_rank_is_threshold_free(self):
        """The reason this API exists: it survives a scale change intact.

        exp30b measured FI deltas spanning 0.004-0.79 on one model and found no
        single threshold that classified three probes of it sensibly, while the
        ordering was stable. Scaling every delta must not move the ranking.
        """
        deltas = {0: 0.004, 1: 0.79, 2: 0.21, 3: 0.08}
        ranked = classify_layers_by_rank(deltas, k=4)
        scaled = classify_layers_by_rank({k: v * 1000 for k, v in deltas.items()}, k=4)
        tiny = classify_layers_by_rank({k: v / 1000 for k, v in deltas.items()}, k=4)
        assert ranked == scaled == tiny == [1, 2, 3, 0]

    def test_empty_input(self):
        assert classify_layers_by_rank({}, k=3) == []
