"""End-to-end test of scripts/run_sal_qat_prototype.py.

Runs ``main()`` on CPU against a 2-layer random-weight ViT classifier and
synthetic images. Only the model loader and the data builder are faked. The
QAT setup and strip, SALTrainer, slicing, quantization, scoring, figures, JSON
and table are the code that will run on the GPU.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sal_qat_prototype.py"
IMG, PATCH, NL, NH, HS, NC = 32, 16, 2, 6, 48, 3


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("run_sal_qat_prototype", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["run_sal_qat_prototype"] = m
    spec.loader.exec_module(m)
    return m


def _tiny_model(num_classes=NC):
    from transformers import ViTConfig, ViTForImageClassification
    return ViTForImageClassification(ViTConfig(
        hidden_size=HS, num_hidden_layers=NL, num_attention_heads=NH,
        intermediate_size=HS * 2, image_size=IMG, patch_size=PATCH,
        num_labels=num_classes))


def _fake_data(sizes, seed, workers=0):
    g = torch.Generator().manual_seed(seed)

    def split(n):
        y = torch.arange(n) % NC
        x = (y.view(-1, 1, 1, 1) * 80 + torch.randint(0, 40, (n, 3, IMG, IMG), generator=g))
        return x.clamp(0, 255).to(torch.uint8), y

    return split(24), split(12), NC


@pytest.fixture
def patched(mod, monkeypatch):
    monkeypatch.setattr(mod, "load_model", _tiny_model)
    monkeypatch.setattr(mod, "build_data", _fake_data)
    monkeypatch.setattr(mod, "IMAGE_SIZE", IMG)
    return mod


# ------------------------------------------------------------------ helpers
def test_fake_quantize_uses_at_most_2_pow_bits_levels(mod):
    w = torch.randn(8, 32)
    for bits in (4, 8):
        q = mod.fake_quantize_weight(w, bits)
        for row in q:
            assert row.unique().numel() <= 2 ** bits
    assert (mod.fake_quantize_weight(w, 8) - w).abs().max() < (mod.fake_quantize_weight(w, 4) - w).abs().max()


def test_quantize_weights_skips_classifier_and_leaves_source(mod):
    m = _tiny_model()
    before = m.classifier.weight.clone()
    q = mod.quantize_weights(m, 4)
    assert torch.equal(q.classifier.weight, before)
    src, dst = dict(m.named_modules()), dict(q.named_modules())
    body = [n for n, x in src.items() if mod._is_quantizable(n, x)]
    assert body
    for n in body:
        assert not torch.equal(src[n].weight, dst[n].weight)


@pytest.mark.parametrize("manual", [False, True])
def test_qat_trains_and_strips_to_plain_linear(mod, manual):
    m = _tiny_model()
    m, backend = mod.setup_qat(m, 4, force_manual=manual)
    assert backend == ("manual-ste" if manual else "torch.ao")
    x = torch.randn(2, 3, IMG, IMG)
    m(pixel_values=x).logits.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)     # STE lets gradients through

    m = mod.strip_qat(m)
    linears = [x for x in m.modules() if isinstance(x, nn.Linear)]
    assert linears and all(type(x) is nn.Linear for x in linears)
    assert not any(hasattr(x, "activation_post_process") for x in m.modules())
    small = mod.sliced(m.eval(), 0.5, 0, {"pixel_values": x})   # slice_heads accepts it
    assert small.config.num_attention_heads == NH // 2


def test_size_mb_shrinks_with_bits(mod):
    m = _tiny_model()
    assert mod.size_mb(m, 4) < mod.size_mb(m, 8) < mod.size_mb(m, None)


def test_uniform_heads_same_count_per_layer_and_deterministic(mod):
    a = mod.uniform_random_heads(4, 12, 0.5, 7)
    assert a == mod.uniform_random_heads(4, 12, 0.5, 7)
    assert all(sum(1 for l, _ in a if l == layer) == 6 for layer in range(4))


# ------------------------------------------------------------------ end to end
def test_full_run_writes_table_json_and_figures(patched, tmp_path):
    payload = patched.main(["--epochs", "1", "--output", str(tmp_path),
                            "--batch-size", "8", "--prune-seeds", "2",
                            "--no-save-models", "--workers", "0"])
    results = payload["results"]
    assert set(results) == {"baseline", "sal", "qat", "sal_qat"}
    for r in results.values():
        for cell in patched.CONFIGS:
            assert 0.0 <= r[cell]["accuracy"] <= 1.0
        assert len(r["prune50"]["per_seed"]) == 2
    assert payload["training"]["sal"]["masker_stats"]["pruned_heads"] > 0
    assert payload["training"]["baseline"]["masker_stats"] is None
    assert payload["training"]["qat"]["qat_backend"] == "torch.ao"
    assert payload["budget"]["int4"]["latency_cpu_ms"] is None

    table = (tmp_path / "summary_table.txt").read_text(encoding="utf-8")
    assert "SAL + QAT" in table and "P50+I4" in table and "verdict" in table
    saved = json.loads((tmp_path / "sal_qat_prototype.json").read_text())
    assert saved["results"]["sal_qat"]["prune50_int4"]["accuracy"] == \
        results["sal_qat"]["prune50_int4"]["accuracy"]
    pytest.importorskip("matplotlib")
    assert (tmp_path / "figures" / "accuracy_by_compression.png").exists()
    assert (tmp_path / "figures" / "size_vs_accuracy.png").exists()


def test_smoke_skips_int4_and_reuses_checkpoints(patched, tmp_path):
    args = ["--smoke", "--output", str(tmp_path), "--batch-size", "8",
            "--variants", "baseline", "--workers", "0"]
    payload = patched.main(args)
    r = payload["results"]["baseline"]
    assert "int8" in r and "int4" not in r and "prune50_int4" not in r
    assert (tmp_path / "models" / "baseline.pt").exists()

    again = patched.main(args + ["--reuse-models"])
    assert "reused_checkpoint" in again["training"]["baseline"]
    assert again["results"]["baseline"]["fp16"]["accuracy"] == r["fp16"]["accuracy"]
