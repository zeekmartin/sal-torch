"""Tests for DriftMonitor / DriftReport (CPU, tiny model fixture)."""
import json

import numpy as np
import pytest
import torch

from sal.drift import DriftMonitor, DriftReport, StructuralSnapshot, compare_snapshots


def _train(model, steps=25, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        inp = torch.randint(0, 100, (4, 16))
        out = model(input_ids=inp, labels=inp)
        out.loss.backward(); opt.step(); opt.zero_grad()


class TestSnapshot:
    def test_snapshot(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        snap = mon.snapshot("s0")
        assert isinstance(snap, StructuralSnapshot)
        assert snap.num_layers == 4 and snap.num_heads == 8
        assert 0.0 <= snap.fi_score <= 1.0
        assert set(snap.layer_map.keys()) <= {0, 1, 2, 3}
        assert len(snap.layer_reps) == 4
        assert "s0" in mon.snapshots

    def test_head_signature_shape(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        snap = mon.snapshot("s0")
        sig = snap.head_signature(0, 0)
        assert sig.shape[1] == 64 // 8  # head_dim


class TestCompare:
    def test_compare_identical(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("a")
        drift = mon.compare("a", "a")
        assert isinstance(drift, DriftReport)
        assert drift.forgetting_score == pytest.approx(0.0, abs=1e-6)
        assert drift.structural_delta == pytest.approx(0.0, abs=1e-6)
        for v in drift.layer_drift.values():
            assert v == pytest.approx(1.0, abs=1e-4)
        assert drift.classification_changes == {}

    def test_compare_after_training(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("before")
        _train(tiny_model, steps=25)
        mon.snapshot("after")
        drift = mon.compare("before", "after")
        assert drift.forgetting_score > 0.0
        # at least one layer's representation moved away from identity
        assert min(drift.layer_drift.values()) < 1.0

    def test_protected_integrity(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("a")
        # without a protected set, protected_integrity is None
        assert mon.compare("a", "a").protected_integrity is None
        # with a protected set, it is a number near 1.0 for the identical case
        d = mon.compare("a", "a", protected_heads=[(0, 0), (1, 1)])
        assert d.protected_integrity == pytest.approx(1.0, abs=1e-4)

    def test_missing_snapshot_raises(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("a")
        with pytest.raises(KeyError):
            mon.compare("a", "nope")


class TestMultipleSnapshots:
    def test_multiple_snapshots(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("t0")
        _train(tiny_model, steps=15, seed=1)
        mon.snapshot("t1")
        _train(tiny_model, steps=15, seed=2)
        mon.snapshot("t2")
        assert set(mon.snapshots.keys()) == {"t0", "t1", "t2"}
        # any pair is comparable
        for a, b in [("t0", "t1"), ("t1", "t2"), ("t0", "t2")]:
            drift = mon.compare(a, b)
            assert 0.0 <= drift.forgetting_score <= 1.0
        # cumulative drift t0->t2 should be >= incremental t0->t1
        assert mon.compare("t0", "t2").forgetting_score >= 0.0


class TestSave:
    def test_drift_report_save(self, tiny_model, probe_data, tmp_path):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("before")
        _train(tiny_model, steps=20)
        mon.snapshot("after")
        drift = mon.compare("before", "after")
        out = tmp_path / "drift.json"
        drift.save(str(out))
        data = json.loads(out.read_text())
        assert "forgetting_score" in data
        assert "structural_delta" in data
        assert "layer_drift" in data
        assert "summary" in data

    def test_drift_report_pdf(self, tiny_model, probe_data, tmp_path):
        pytest.importorskip("matplotlib"); pytest.importorskip("fpdf")
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        mon.snapshot("before")
        _train(tiny_model, steps=20)
        mon.snapshot("after")
        drift = mon.compare("before", "after")
        out = tmp_path / "drift.pdf"
        drift.save(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_snapshot_roundtrip(self, tiny_model, probe_data):
        mon = DriftMonitor(tiny_model, probe_data, num_samples=40)
        snap = mon.snapshot("a")
        d = snap.to_dict()
        snap2 = StructuralSnapshot.from_dict(d)
        drift = compare_snapshots(snap, snap2)
        assert drift.forgetting_score == pytest.approx(0.0, abs=1e-4)
