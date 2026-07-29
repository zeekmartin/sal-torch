"""Tests for the robustness suite (CPU, tiny model fixture)."""
import json

import pytest
import torch
import torch.nn as nn

from sal.robustness import (DEFAULT_METHODS, MethodRobustness, RobustnessReport,
                            RobustnessTest, _degradation, _ffn_expansion_linears,
                            _NeuronDropout, _quantize_int4_simulated, _quantize_int8,
                            robustness_compare)

try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# --------------------------------------------------------------------- helpers
def _data(n_batches=5, bs=4, sl=16, seed=0):
    """Constant-token sequences: the model can learn these, so accuracy is meaningful."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        ids = torch.randint(0, 4, (bs, 1), generator=g).repeat(1, sl)
        out.append({"input_ids": ids, "labels": ids.clone()})
    return out


def _train(model, data, steps=90, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        out = model(**data[s % len(data)])
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    return model


@pytest.fixture
def trained_model(tiny_model):
    torch.manual_seed(0)
    return _train(tiny_model, _data())


def _batch(bs=4, sl=16):
    return {"input_ids": torch.zeros(bs, sl, dtype=torch.long)}


# ---------------------------------------------------------------- quantization
def test_int8_quantization(tiny_model):
    clean_logits = tiny_model(**_batch()).logits
    q = _quantize_int8(tiny_model)
    q_logits = q(**_batch()).logits
    assert q_logits.shape == clean_logits.shape
    assert torch.isfinite(q_logits).all()
    # The original model must be untouched.
    assert isinstance(tiny_model.head, nn.Linear)


def test_int8_method_runs(trained_model):
    test = RobustnessTest(trained_model, _data(), metric="accuracy", batch_size=4)
    report = test.run(["int8"])
    r = report.results[0]
    assert not r.skipped
    assert r.backend == "torch-dynamic-int8"
    assert 0.0 <= r.after <= 1.0


def test_int4_simulated_changes_weights_but_keeps_shape(tiny_model):
    before = tiny_model.head.weight.detach().clone()
    q = _quantize_int4_simulated(tiny_model)
    assert q.head.weight.shape == before.shape
    assert not torch.allclose(q.head.weight, before)          # rounding happened
    assert torch.allclose(tiny_model.head.weight, before)     # original untouched
    assert torch.isfinite(q(**_batch()).logits).all()


def test_int4_falls_back_to_simulation_without_bitsandbytes(trained_model):
    report = RobustnessTest(trained_model, _data(), batch_size=4).run(["int4"])
    r = report.results[0]
    assert not r.skipped
    if HAS_BNB and torch.cuda.is_available():
        assert r.backend == "bitsandbytes-nf4"
    else:
        assert r.backend == "simulated-int4"
        assert "simulated" in r.note


def test_int4_skips_when_simulation_disallowed(trained_model):
    test = RobustnessTest(trained_model, _data(), batch_size=4, allow_simulated_quant=False)
    r = test.run(["int4"]).results[0]
    if HAS_BNB and torch.cuda.is_available():
        assert not r.skipped
    else:
        assert r.skipped
        assert "bitsandbytes" in r.note


@pytest.mark.skipif(not HAS_BNB, reason="needs bitsandbytes")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 4-bit needs CUDA")
def test_int4_bitsandbytes_backend(trained_model):
    r = RobustnessTest(trained_model.cuda(), _data(), batch_size=4).run(["int4"]).results[0]
    assert r.backend == "bitsandbytes-nf4"
    assert not r.skipped


# ---------------------------------------------------------------- head pruning
def test_head_pruning_methods(trained_model):
    test = RobustnessTest(trained_model, _data(), metric="accuracy", batch_size=4)
    report = test.run(["head_pruning_33", "head_pruning_50"])
    assert [r.method for r in report.results] == ["head_pruning_33", "head_pruning_50"]
    for r in report.results:
        assert not r.skipped
        assert 0.0 <= r.after <= 1.0
        assert r.time_seconds >= 0.0


def test_head_pruning_leaves_no_hooks_behind(trained_model):
    before = RobustnessTest(trained_model, _data(), batch_size=4).baseline()
    RobustnessTest(trained_model, _data(), batch_size=4).run(["head_pruning_50"])
    after = RobustnessTest(trained_model, _data(), batch_size=4).baseline()
    assert before == pytest.approx(after)


def test_head_pruning_rejects_out_of_range(trained_model):
    r = RobustnessTest(trained_model, _data(), batch_size=4).run(["head_pruning_100"]).results[0]
    assert r.skipped and "failed" in r.note


# -------------------------------------------------------------- neuron dropout
def test_ffn_expansion_detection(tiny_model):
    mods = _ffn_expansion_linears(tiny_model)
    assert len(mods) == 4                                  # one per layer
    for name, m in mods:
        assert m.out_features > m.in_features
        assert "mlp" in name


def test_neuron_dropout_changes_output(tiny_model):
    batch = _batch()
    clean = tiny_model(**batch).logits
    mods = _ffn_expansion_linears(tiny_model)
    with _NeuronDropout(mods, p=0.5, seed=0):
        dropped = tiny_model(**batch).logits
    assert dropped.shape == clean.shape
    assert not torch.allclose(clean, dropped)
    # The hooks must come off again.
    assert torch.allclose(tiny_model(**batch).logits, clean)


def test_neuron_dropout_zero_fraction_is_a_noop(tiny_model):
    batch = _batch()
    clean = tiny_model(**batch).logits
    with _NeuronDropout(_ffn_expansion_linears(tiny_model), p=0.0, seed=0):
        assert torch.allclose(tiny_model(**batch).logits, clean)


def test_neuron_dropout_method_reports_std(trained_model):
    test = RobustnessTest(trained_model, _data(), metric="accuracy", batch_size=4,
                          dropout_trials=3)
    r = test.run(["neuron_dropout_20"]).results[0]
    assert not r.skipped
    assert r.std is not None and r.std >= 0.0
    assert "3 trial(s)" in r.note


# ---------------------------------------------------------------------- report
def test_robustness_report(trained_model, tmp_path):
    test = RobustnessTest(trained_model, _data(), metric="accuracy", batch_size=4,
                          dropout_trials=2, model_name="tiny")
    report = test.run(["int8", "head_pruning_33", "neuron_dropout_10"])

    table = report.table
    assert "method" in table and "survived" in table
    for name in ("int8", "head_pruning_33", "neuron_dropout_10"):
        assert name in table
    assert "robustness_score=" in table
    assert "tiny" in report.summary

    assert 0.0 <= report.robustness_score <= 1.0
    assert 0.0 <= report.survival_rate <= 1.0

    out = tmp_path / "robustness.json"
    report.save(str(out))
    data = json.loads(out.read_text())
    assert data["metric"] == "accuracy"
    assert len(data["results"]) == 3
    assert 0.0 <= data["robustness_score"] <= 1.0
    assert data["results"][0]["backend"] == "torch-dynamic-int8"


def test_report_save_rejects_unknown_suffix(trained_model, tmp_path):
    report = RobustnessTest(trained_model, _data(), batch_size=4).run(["int8"])
    with pytest.raises(ValueError, match="Unsupported"):
        report.save(str(tmp_path / "robustness.txt"))


def test_robustness_pdf(trained_model, tmp_path):
    pytest.importorskip("matplotlib"); pytest.importorskip("fpdf")
    report = RobustnessTest(trained_model, _data(), batch_size=4, dropout_trials=1).run(
        ["int8", "int4", "head_pruning_33", "neuron_dropout_10"])
    out = tmp_path / "robustness.pdf"
    report.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_unknown_method_is_recorded_not_raised(trained_model):
    r = RobustnessTest(trained_model, _data(), batch_size=4).run(["not_a_method"]).results[0]
    assert r.skipped and "Unknown method" in r.note


def test_invalid_metric_and_threshold(trained_model):
    with pytest.raises(ValueError, match="metric"):
        RobustnessTest(trained_model, _data(), metric="f1")
    with pytest.raises(ValueError, match="survival_threshold"):
        RobustnessTest(trained_model, _data(), survival_threshold=1.5)


def test_default_methods_constant():
    assert DEFAULT_METHODS == ("int8", "int4", "head_pruning_33", "head_pruning_50",
                               "neuron_dropout_10", "neuron_dropout_20")


# ------------------------------------------------------------ survival logic
def _row(baseline, after, threshold, direction="higher_better"):
    survived = _degradation(baseline, after, direction) <= threshold + 1e-9
    return MethodRobustness(method="m", baseline=baseline, after=after, survived=survived,
                            time_seconds=0.0, metric_direction=direction)


def test_survival_threshold():
    # 5% default: a 3% relative drop survives, a 12% drop does not.
    assert _row(0.900, 0.873, 0.05).survived
    assert not _row(0.900, 0.792, 0.05).survived
    # Exactly at the threshold counts as survived.
    assert _row(1.000, 0.950, 0.05).survived
    # Improvements always survive.
    assert _row(0.900, 0.950, 0.05).survived
    # A looser threshold rescues the same measurement.
    assert _row(0.900, 0.792, 0.20).survived


def test_survival_threshold_for_loss_metric():
    # Lower loss is better, so a rise in loss is the degradation.
    assert _row(1.00, 1.02, 0.05, "lower_better").survived
    assert not _row(1.00, 1.30, 0.05, "lower_better").survived


def test_degradation_and_retention():
    r = _row(0.800, 0.720, 0.05)
    assert r.delta == pytest.approx(-0.08)
    assert r.degradation == pytest.approx(0.10)
    assert r.retention == pytest.approx(0.90)
    # Retention is clipped, so a total collapse cannot drag the score negative.
    assert _row(0.800, -1.0, 0.05).retention == 0.0


def test_degradation_handles_zero_baseline():
    # No relative scale at zero — fall back to the absolute change, not a blow-up.
    assert _degradation(0.0, 0.1, "higher_better") == pytest.approx(-0.1)


def test_robustness_score_ignores_skipped():
    rows = [MethodRobustness("a", 1.0, 0.9, True, 0.0),
            MethodRobustness("b", 1.0, float("nan"), False, 0.0, skipped=True)]
    report = RobustnessReport(results=rows, metric="accuracy", survival_threshold=0.05)
    assert len(report.evaluated) == 1
    assert report.robustness_score == pytest.approx(0.9)
    assert report.survival_rate == pytest.approx(1.0)
    assert "skipped" in report.table


# --------------------------------------------------------------------- compare
def test_compare(tiny_model, tmp_path):
    torch.manual_seed(0)
    data = _data()
    sal_model = _train(tiny_model, data)

    from conftest import TinyTransformer  # a second, independently trained model
    torch.manual_seed(1)
    baseline_model = _train(TinyTransformer(), data)

    methods = ["int8", "head_pruning_33", "neuron_dropout_10"]
    result = robustness_compare(sal_model, baseline_model, data, methods=methods,
                                metric="accuracy", batch_size=4, dropout_trials=1)

    assert [r.method for r in result.rows] == methods
    for row in result.rows:
        assert row.winner in ("SAL", "baseline", "tie", "n/a")
    assert "method" in result.table and "winner" in result.table
    assert "compression methods" in result.summary or "nothing to compare" in result.summary
    assert 0 <= result.sal_wins <= len(result.comparable)

    out = tmp_path / "comparison.json"
    result.save(str(out))
    data_json = json.loads(out.read_text())
    assert len(data_json["rows"]) == len(methods)
    assert "baseline_report" in data_json and "sal_report" in data_json


def test_compare_pdf(tiny_model, tmp_path):
    pytest.importorskip("matplotlib"); pytest.importorskip("fpdf")
    from conftest import TinyTransformer
    torch.manual_seed(0)
    data = _data()
    result = robustness_compare(_train(tiny_model, data), _train(TinyTransformer(), data),
                                data, methods=["int8", "head_pruning_33"],
                                metric="accuracy", batch_size=4)
    out = tmp_path / "robustness_comparison.pdf"
    result.save(str(out))
    assert out.exists() and out.stat().st_size > 0
