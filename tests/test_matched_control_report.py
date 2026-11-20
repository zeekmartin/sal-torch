"""A cost is a cost only against a control trained on the arm's own budget.

Everything here runs the report: the properties are read off a pipeline that
actually compressed and recovered a model, or off a CompressionReport built
with the stages a run would have produced. Nothing asserts on source text.

``tests/test_matched_control.py`` pins the same rule in the experiment harness
(``experiments/matched_control.py``); this file pins it in the product, where
it is stricter: the harness reports a delta against the pretrained checkpoint
beside the gain for readers who know the convention, and the product reports
none at all.
"""
import json

import pytest
import torch

from sal.pipeline import CompressionPipeline, CompressionReport, Stage
from sal.recovery import RecoveryControl


def _data(n=4, bs=4, sl=16):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(0, 100, (bs, sl)),
             "labels": torch.randint(0, 100, (bs, sl))} for _ in range(n)]


def _report(metric="accuracy", pretrained=0.78, arm=0.77, control=None,
            recovery=None):
    """The stages a run produces, without spending a minute producing them."""
    stages = [Stage(name="original", size_mb=100.0, accuracy=pretrained),
              Stage(name="pruned+sliced", size_mb=84.0, accuracy=0.10),
              Stage(name="recovered", size_mb=84.0, accuracy=arm)]
    return CompressionReport(stages=stages, metric=metric, control=control,
                             recovery=recovery or {"epochs": 2})


# ----------------------------------------------------------- with a control
def test_cost_and_gain_with_a_matched_control():
    r = _report(pretrained=0.7844, arm=0.78816,
                control=RecoveryControl(0.81554, epochs=2, seeds=3))
    assert r.budget_matched is True
    assert r.pruning_cost_pp == pytest.approx(-2.738, abs=1e-3)
    assert r.fine_tuning_gain_pp == pytest.approx(3.114, abs=1e-3)
    assert "pruning cost -2.74pp" in r.cost_statement
    assert "same epochs" in r.cost_statement


def test_gain_is_exactly_control_minus_pretrained():
    r = _report(pretrained=0.5, arm=0.9, control=RecoveryControl(0.8, epochs=2))
    assert r.fine_tuning_gain == pytest.approx(0.3)
    assert r.fine_tuning_gain_pp == pytest.approx(30.0)
    assert r.pruning_cost == pytest.approx(0.1)        # arm minus control


def test_cost_is_absent_when_the_epoch_counts_differ():
    r = _report(control=RecoveryControl(0.81554, epochs=5, seeds=3))
    assert r.budget_matched is False
    assert r.pruning_cost is None and r.pruning_cost_pp is None
    assert "not isolated" in r.cost_statement
    assert "different budget" in r.cost_statement
    assert "epochs=5" in r.cost_statement and "epochs=2" in r.cost_statement


def test_a_token_budget_matches_only_another_token_budget():
    rec = {"epochs": None, "tokens": 32_000_000}
    matched = _report(metric="loss", pretrained=14.6240, arm=16.2578,
                      control=RecoveryControl(15.2639, tokens=32_000_000,
                                              seeds=3, metric="loss"),
                      recovery=rec)
    assert matched.budget_matched is True
    # Lower is better, so the sign flips: a positive cost means worse.
    assert matched.pruning_cost == pytest.approx(0.9939, abs=1e-4)
    assert matched.pruning_cost_pp is None, "perplexity is not points of accuracy"
    assert "pruning cost +0.9939 loss" in matched.cost_statement
    # The corpus itself cost the unpruned model 0.6399 of perplexity.
    assert matched.fine_tuning_gain == pytest.approx(0.6399, abs=1e-4)
    assert "the budget itself cost +0.6399 loss" in matched.cost_statement

    half = _report(metric="loss", pretrained=14.6240, arm=16.2578,
                   control=RecoveryControl(15.2639, tokens=16_000_000,
                                           metric="loss"),
                   recovery=rec)
    assert half.budget_matched is False and half.pruning_cost is None

    epochs_against_tokens = _report(
        metric="loss", control=RecoveryControl(15.2639, epochs=2,
                                               metric="loss"),
        recovery=rec)
    assert epochs_against_tokens.budget_matched is False


# -------------------------------------------------------- without a control
def test_without_a_control_there_is_no_cost_and_no_pretrained_delta():
    r = _report(pretrained=0.7844, arm=0.78816)
    assert r.control is None and r.budget_matched is None
    assert r.pruning_cost is None and r.pruning_cost_pp is None
    assert r.fine_tuning_gain is None and r.fine_tuning_gain_pp is None
    assert r.NO_CONTROL in r.cost_statement

    d = r.to_dict()
    for field in ("pruning_cost", "pruning_cost_pp", "fine_tuning_gain",
                  "fine_tuning_gain_pp"):
        assert field not in d, f"{field} must be absent, not null"
    assert d["control"] is None and d["budget_matched"] is None
    assert "not isolated" in d["cost_statement"]
    json.loads(json.dumps(d))


def test_a_recovered_summary_does_not_present_a_pretrained_delta_as_a_cost():
    """After a recovery, last minus first is the sum of two effects."""
    r = _report(pretrained=0.7844, arm=0.78816)
    assert r.recovered is True
    assert "relative" not in r.summary, "a relative delta reads as a cost"
    assert "after recovery" in r.summary
    assert r.NO_CONTROL in r.summary
    # The waterfall numbers are still there; it is the framing that changed.
    assert "0.784" in r.summary and "0.788" in r.summary


def test_a_one_shot_summary_still_reports_its_own_damage():
    """With no recovery there is no second effect, so the delta is the cost."""
    stages = [Stage(name="original", size_mb=100.0, accuracy=0.78),
              Stage(name="pruned+sliced", size_mb=84.0, accuracy=0.10)]
    r = CompressionReport(stages=stages, metric="accuracy")
    assert r.recovered is False
    assert "one-shot" in r.summary
    assert r.accuracy_delta == pytest.approx(-0.68)


# ------------------------------------------------------- through a real run
def test_the_pipeline_carries_the_control_into_the_report():
    data = _data()
    pipe = CompressionPipeline(TinyModel(), data, metric="loss", batch_size=4)
    pipe.compress(pruning=0.25, quantization=None)
    control = RecoveryControl(pipe.original_accuracy, epochs=1, metric="loss",
                              seeds=3)
    pipe.recover(data, domain="same as eval", epochs=1, control=control)
    r = pipe.validate()
    assert r.control is control
    assert r.budget_matched is True
    assert r.pruning_cost == pytest.approx(
        r.arm_accuracy - control.accuracy, abs=1e-9)
    assert "pruning cost" in r.to_dict()["cost_statement"]
    assert "pruning_cost" in r.to_dict()


def test_the_pipeline_without_a_control_says_so(tmp_path):
    data = _data()
    pipe = CompressionPipeline(TinyModel(), data, metric="loss", batch_size=4)
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(data, domain="same as eval", epochs=1)
    r = pipe.validate()
    assert r.pruning_cost is None
    assert r.NO_CONTROL in r.cost_statement
    out = tmp_path / "report.json"
    r.save(str(out))
    d = json.loads(out.read_text())
    assert "pruning_cost" not in d and "not isolated" in d["cost_statement"]


def TinyModel():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import TinyTransformer
    torch.manual_seed(42)
    return TinyTransformer()
