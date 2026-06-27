"""Tests for StructuralGuard (CPU, tiny model fixture)."""
import json

import numpy as np
import pytest
import torch

from sal.guard import StructuralGuard, StructuralGuardCallback, _per_head_mi


HD = 64 // 8  # head_dim for the tiny model (hidden 64, 8 heads)


def _q_head_grad(model, layer, head):
    w = model.transformer.h[layer].attn.q_proj.weight.grad
    return float(w[head * HD:(head + 1) * HD, :].abs().sum())


def _o_head_grad(model, layer, head):
    w = model.transformer.h[layer].attn.out_proj.weight.grad
    return float(w[:, head * HD:(head + 1) * HD].abs().sum())


def _backward_once(model, seed=0):
    torch.manual_seed(seed)
    model.zero_grad()
    inp = torch.randint(0, 100, (4, 16))
    out = model(input_ids=inp, labels=inp)
    out.loss.backward()


class TestFromModel:
    def test_from_model(self, tiny_model, probe_data):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        assert g.num_layers == 4 and g.num_heads == 8
        # protected + trainable partition all 32 heads with no overlap.
        assert len(g.protected_heads) + len(g.trainable_heads) == 32
        assert not (set(g.protected_heads) & set(g.trainable_heads))
        assert len(g.protected_heads) == round(0.5 * 32)
        # protection_map keys are layers, values are head-index lists.
        for layer, heads in g.protection_map.items():
            assert 0 <= layer < 4
            for h in heads:
                assert (layer, h) in set(g.protected_heads)

    def test_per_head_mi_range(self, tiny_model, probe_data):
        from sal.drift import take_snapshot
        snap = take_snapshot(tiny_model, probe_data, "b", num_samples=40)
        mi = _per_head_mi(snap.layer_reps, 8)
        assert len(mi) == 32
        for v in mi.values():
            assert 0.0 <= v <= 1.0


class TestProtectionLevel:
    def test_zero_protects_nothing(self, tiny_model, probe_data):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.0, num_samples=40)
        assert g.protected_heads == []
        assert len(g.trainable_heads) == 32

    def test_one_protects_everything(self, tiny_model, probe_data):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=1.0, num_samples=40)
        assert len(g.protected_heads) == 32
        assert g.trainable_heads == []

    def test_invalid_level(self, tiny_model, probe_data):
        with pytest.raises(ValueError):
            StructuralGuard.from_model(tiny_model, probe_data, protection_level=1.5, num_samples=40)


class TestProtect:
    def test_protect_zeros_gradients(self, tiny_model, probe_data):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        g.protect(tiny_model)
        _backward_once(tiny_model)

        for (l, h) in g.protected_heads:
            assert _q_head_grad(tiny_model, l, h) == 0.0
            assert _o_head_grad(tiny_model, l, h) == 0.0
        # At least some trainable head must carry a non-zero gradient.
        live = [(_q_head_grad(tiny_model, l, h) + _o_head_grad(tiny_model, l, h))
                for (l, h) in g.trainable_heads]
        assert max(live) > 0.0
        g.release()

    def test_release(self, tiny_model, probe_data):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        g.protect(tiny_model)
        assert g.is_active
        g.release()
        assert not g.is_active
        _backward_once(tiny_model)
        # After release, every protected head's gradient flows again.
        flowed = [(_q_head_grad(tiny_model, l, h) + _o_head_grad(tiny_model, l, h))
                  for (l, h) in g.protected_heads]
        assert max(flowed) > 0.0

    def test_protect_is_reentrant(self, tiny_model, probe_data):
        # Calling protect twice should not stack hooks (no double counting).
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        g.protect(tiny_model)
        n1 = len(g._handles)
        g.protect(tiny_model)
        assert len(g._handles) == n1
        g.release()


class TestSaveLoad:
    def test_save_load(self, tiny_model, probe_data, tmp_path):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        out = tmp_path / "guard.json"
        g.save(str(out))
        data = json.loads(out.read_text())
        assert data["protection_level"] == 0.5
        assert data["num_layers"] == 4 and data["num_heads"] == 8

        g2 = StructuralGuard.load(str(out))
        assert g2.protected_heads == g.protected_heads
        assert g2.trainable_heads == g.trainable_heads
        assert g2.protection_map == g.protection_map
        assert g2.protection_level == g.protection_level
        # A reloaded guard can still protect a model.
        g2.protect(tiny_model)
        assert g2.is_active
        g2.release()

    def test_loaded_guard_measures_drift(self, tiny_model, probe_data, tmp_path):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        out = tmp_path / "guard.json"
        g.save(str(out))
        g2 = StructuralGuard.load(str(out))
        drift = g2.measure_drift(tiny_model, probe_dataset=probe_data)
        # Same model -> minimal drift.
        assert drift.forgetting_score < 0.1


class TestComposeWithSAL:
    def test_compose_with_sal(self, tiny_model, tiny_config, probe_data):
        from sal.masker import HeadMasker

        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        g.protect(tiny_model)

        masker = HeadMasker(tiny_model, tiny_config, seed=0)
        masker.install()
        opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        total = 12
        for step in range(total):
            masker.step(step, total)
            inp = torch.randint(0, 100, (4, 16))
            out = tiny_model(input_ids=inp, labels=inp)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        masker.remove()
        g.release()
        # If we got here without raising, SAL + guard compose cleanly.
        drift = g.measure_drift(tiny_model, probe_dataset=probe_data)
        assert 0.0 <= drift.forgetting_score <= 1.0


class TestCallback:
    def test_callback(self, tiny_model, probe_data):
        g = StructuralGuard.from_model(tiny_model, probe_data, protection_level=0.5, num_samples=40)
        cb = StructuralGuardCallback(g)
        cb.on_train_begin(None, None, None, model=tiny_model)
        assert g.is_active
        # simulate a couple of optimizer steps
        opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        for _ in range(5):
            inp = torch.randint(0, 100, (4, 16))
            out = tiny_model(input_ids=inp, labels=inp)
            out.loss.backward(); opt.step(); opt.zero_grad()
        cb.on_train_end(None, None, None, model=tiny_model)
        assert not g.is_active
        assert cb.drift_report is not None
        assert 0.0 <= cb.drift_report.forgetting_score <= 1.0
