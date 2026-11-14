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


def _fake_data(sizes, seed, workers=0, dataset_id=None, cache_dir=None):
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


# ------------------------------------------------------------ control variants
def _results_for(keys, cell_acc=0.5):
    """A minimal results dict covering every scored cell, for table formatting."""
    return {k: {cell: {"accuracy": cell_acc + 0.01 * i}
                for cell in ("fp16", "int8", "int4", "prune50", "prune50_int4")}
            for i, k in enumerate(keys)}


def test_summary_table_handles_three_control_arms_without_sal(mod):
    """The expF regression: three deltas that are not sal/qat/sal_qat.

    ``--variants baseline drophead progressive fi_drophead`` produces exactly
    three non-baseline deltas, so a guard that counts them instead of naming
    them walks into ``deltas["sal"]``. Twenty 35-minute jobs died this way.
    """
    results = _results_for(("baseline", "drophead", "progressive", "fi_drophead"))
    table = mod.summary_table(results, {}, 1, False)
    assert "additive prediction" not in table
    assert "verdict" not in table


def test_summary_table_still_reports_additive_prediction_for_sal_qat(mod):
    results = _results_for(("baseline", "sal", "qat", "sal_qat"))
    table = mod.summary_table(results, {}, 1, False)
    assert "additive prediction" in table and "verdict" in table


def test_metrics_survive_a_broken_summary_table(patched, tmp_path, monkeypatch):
    """Presentation must never be able to destroy a finished run."""
    def boom(*a, **k):
        raise KeyError("sal")

    monkeypatch.setattr(patched, "summary_table", boom)
    payload = patched.main(["--epochs", "1", "--output", str(tmp_path),
                            "--batch-size", "8", "--prune-seeds", "1",
                            "--variants", "baseline", "--no-save-models",
                            "--workers", "0"])
    saved = json.loads((tmp_path / "sal_qat_prototype.json").read_text())
    assert saved["results"]["baseline"]["fp16"]["accuracy"] == \
        payload["results"]["baseline"]["fp16"]["accuracy"]
    assert not (tmp_path / "summary_table.txt").exists()


# --------------------------------------------------------------------------
# Audit 2, rules 1 and 2: one validated probe builder, and a reference gate.
#
# These pin the two failures that cost this campaign the most GPU time. An
# unnormalized probe (exp37, exp42) completes, writes JSON and prints a table
# while measuring a different quantity; an arm scored against the wrong
# control (exp32) is off by about a point with nothing to flag it. Neither is
# a crash, so only an assertion catches either one.
# --------------------------------------------------------------------------

def _uint8(n=64):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (n, 3, IMG, IMG), generator=g,
                         dtype=torch.uint8)


def test_build_probe_normalizes_and_reports_its_geometry(mod):
    probe = mod.build_probe(_uint8(64), 4, 16)
    stats = mod.validate_probe(probe)
    assert stats == {**stats, "batches": 4, "batch_size": 16, "images": 64}
    assert stats["min"] < -0.9 and stats["max"] > 0.9
    assert all(torch.is_floating_point(b["pixel_values"]) for b in probe)


def test_build_probe_is_deterministic_on_a_fixed_block(mod):
    imgs = _uint8(64)
    a = mod.build_probe(imgs, 4, 16)
    b = mod.build_probe(imgs, 4, 16)
    assert all(torch.equal(x["pixel_values"], y["pixel_values"])
               for x, y in zip(a, b))


def test_build_probe_distinguishes_disjoint_blocks(mod):
    imgs = _uint8(64)
    a = mod.build_probe(imgs[:32], 2, 16)
    b = mod.build_probe(imgs[32:], 2, 16)
    assert not torch.equal(a[0]["pixel_values"], b[0]["pixel_values"])


def test_validate_probe_rejects_an_integer_probe(mod):
    """The exp37/exp42 bug: the raw cache handed straight to the scanner."""
    with pytest.raises(ValueError, match="dtype"):
        mod.validate_probe([{"pixel_values": _uint8(16)}])


@pytest.mark.parametrize("scale, label", [(255.0, "[0,1]"), (1.0, "[0,255]")])
def test_validate_probe_rejects_float_but_unnormalized(mod, scale, label):
    """Float dtype is not enough: a normalized probe straddles zero."""
    with pytest.raises(ValueError, match="unnormalized"):
        mod.validate_probe([{"pixel_values": _uint8(16).float() / scale}])


def test_validate_probe_rejects_ragged_batches(mod):
    """num_samples and batch_size are recorded; ragged makes them fiction."""
    imgs = _uint8(24)
    with pytest.raises(ValueError, match="ragged"):
        mod.validate_probe([{"pixel_values": mod.normalize(imgs[:16])},
                            {"pixel_values": mod.normalize(imgs[16:])}])


def test_validate_probe_rejects_non_finite_values(mod):
    bad = torch.tensor([[[[float("nan")]]], [[[-1.0]]]])
    with pytest.raises(ValueError, match="non-finite"):
        mod.validate_probe([{"pixel_values": bad}])


def test_validate_probe_rejects_a_malformed_probe(mod):
    with pytest.raises(ValueError, match="non-empty"):
        mod.validate_probe([])
    with pytest.raises(ValueError, match="pixel_values"):
        mod.validate_probe([{"input_ids": torch.zeros(2, 2)}])


def test_build_probe_refuses_to_short_change_the_probe(mod):
    """Silently returning 4 batches when 32 were asked for would move FI."""
    with pytest.raises(ValueError, match="needs 512 images"):
        mod.build_probe(_uint8(64), 32, 16)


def test_check_reference_accepts_the_published_value(mod):
    seen = []
    mod.check_reference("vit_b16_dense_fi_500img", 0.070874, seen.append)
    assert seen and "OK" in seen[0]


def test_the_dense_fi_band_catches_the_failures_it_is_for(mod):
    """The band's job is an order-of-magnitude probe error, not sampling noise.

    A min/max band over the measured draws was tried first and fired on a
    legitimate reading one edge outside the previous extreme. These are the
    four numbers that decide the width: two real readings that must pass, and
    the two measured wrong-probe readings that must fail.
    """
    ok = lambda v: mod.check_reference(                      # noqa: E731
        "vit_b16_dense_fi_band", v, lambda _: None)
    ok(0.064078)        # the reading the first, too-tight band rejected
    ok(0.082524)        # the highest normalized draw measured
    for bad, why in ((0.132039, "the unnormalized uint8 run"),
                     (0.151456, "a 64-image probe")):
        with pytest.raises(RuntimeError):
            ok(bad)


def test_check_reference_refuses_another_probes_reading(mod):
    """0.066990 is a real dense FI -- from probe 1, not probe 0."""
    with pytest.raises(RuntimeError) as e:
        mod.check_reference("vit_b16_dense_fi_500img", 0.066990, lambda _: None)
    assert "EXP42_FI_THRESHOLDS.md" in str(e.value)


def test_check_reference_admits_one_seed_inside_the_published_spread(mod):
    mod.check_reference("vit_b16_in1k_control_2ep_top1", 0.81676,
                        lambda _: None)
    with pytest.raises(RuntimeError):
        mod.check_reference("vit_b16_in1k_control_2ep_top1", 0.80556,
                            lambda _: None)


def test_check_reference_refuses_an_unknown_key(mod):
    with pytest.raises(KeyError):
        mod.check_reference("no_such_key", 1.0, lambda _: None)


def test_every_reference_value_names_its_source(mod):
    """Cardinal rule 7: a quotable number names the page it came from."""
    for key, ref in mod.REFERENCE_VALUES.items():
        assert set(ref) >= {"source", "note"}, key
        assert ref["source"].endswith(".md"), key
        # Two shapes: a point with a tolerance, or a band. A probe-dependent
        # value only has a band -- the spread IS the measurement there, so a
        # point gate on one would assert a precision it does not have.
        if "lo" in ref:
            assert ref["lo"] < ref["hi"], key
            assert "value" not in ref and "tol" not in ref, key
        else:
            assert {"value", "tol"} <= set(ref), key
            assert ref["tol"] >= 0.0, key
