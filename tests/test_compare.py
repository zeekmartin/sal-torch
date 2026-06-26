"""Tests for sal.compare() (CPU, tiny model fixture)."""
import json

import pytest
import torch

from sal.compare import CompareResult, compare, register_method


def _labeled_data(n_batches=5, bs=4, sl=16):
    torch.manual_seed(0)
    # Tiny model is a causal LM, so labels == input_ids and metric is "loss".
    return [{"input_ids": torch.randint(0, 100, (bs, sl)),
             "labels": torch.randint(0, 100, (bs, sl))} for _ in range(n_batches)]


class TestPosthocMethods:
    def test_magnitude_and_random(self, tiny_model):
        data = _labeled_data()
        res = compare(tiny_model, data, data, methods=["magnitude", "random_posthoc"],
                      compression=0.33, metric="loss")
        assert isinstance(res, CompareResult)
        assert {r.method for r in res.results} == {"magnitude", "random_posthoc"}
        for r in res.results:
            assert r.pruned_heads == round(0.33 * 32)
            assert r.score == r.score  # finite (not NaN)
            assert r.time_seconds >= 0.0

    def test_winner_loss_is_min(self, tiny_model):
        data = _labeled_data()
        res = compare(tiny_model, data, data, methods=["magnitude", "random_posthoc"], metric="loss")
        assert res.winner == min(res.results, key=lambda r: r.score).method

    def test_compression_controls_pruned_count(self, tiny_model):
        data = _labeled_data()
        res = compare(tiny_model, data, data, methods=["random_posthoc"], compression=0.5, metric="loss")
        assert res.results[0].pruned_heads == round(0.5 * 32)


class TestTableAndSave:
    def test_table_is_string(self, tiny_model):
        data = _labeled_data()
        res = compare(tiny_model, data, data, methods=["magnitude", "random_posthoc"], metric="loss")
        assert isinstance(res.table, str)
        assert "method" in res.table and "pruned_heads" in res.table

    def test_save_json(self, tiny_model, tmp_path):
        data = _labeled_data()
        res = compare(tiny_model, data, data, methods=["magnitude", "random_posthoc"], metric="loss")
        out = tmp_path / "comparison.json"
        res.save(str(out))
        payload = json.loads(out.read_text())
        assert payload["metric"] == "loss"
        assert "winner" in payload and len(payload["results"]) == 2


class TestPlugin:
    def test_register_custom_method(self, tiny_model):
        seen = {}

        def my_method(model, dataset, eval_dataset, ctx):
            seen["ctx_keys"] = set(ctx.keys())
            return 0.123

        register_method("dummy", my_method)
        data = _labeled_data()
        res = compare(tiny_model, data, data, methods=["dummy"], metric="loss")
        assert seen, "custom method was not called"
        assert res.results[0].method == "dummy"
        assert res.results[0].score == pytest.approx(0.123)

    def test_register_via_attribute(self):
        assert compare.register_method is register_method


class TestValidation:
    def test_invalid_metric(self, tiny_model):
        with pytest.raises(ValueError):
            compare(tiny_model, _labeled_data(), methods=["magnitude"], metric="bogus")

    def test_unknown_method(self, tiny_model):
        with pytest.raises(ValueError):
            compare(tiny_model, _labeled_data(), methods=["does_not_exist"], metric="loss")


@pytest.mark.integration
def test_compare_sal(tiny_model):
    data = _labeled_data()
    res = compare(tiny_model, data, data, methods=["sal", "random_posthoc"],
                  compression=0.33, sal_epochs=1, metric="loss")
    assert {r.method for r in res.results} == {"sal", "random_posthoc"}
    sal = next(r for r in res.results if r.method == "sal")
    assert sal.score == sal.score  # finite
