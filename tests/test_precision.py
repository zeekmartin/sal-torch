"""The load-time precision check, run against real loads.

Every test here builds a model, performs or inspects a real load, and reads
what the check returns. The failure it exists for is silent: ``load_state_dict``
narrows an fp32 checkpoint into fp16 parameters without a word, which is how
the 6.7B zero-shot pass of this project's campaign was measured in half
precision and had to be re-run.
"""
import logging

import pytest
import torch
import torch.nn as nn

from sal.precision import (PrecisionError, check_load_precision,
                           model_precision, narrower)


class Config:
    def __init__(self, torch_dtype=None):
        self.torch_dtype = torch_dtype


def _model(dtype=torch.float32, config_dtype=None):
    m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4)).to(dtype)
    m.config = Config(config_dtype)
    return m


# ------------------------------------------------------------------ the order
def test_narrower_knows_what_narrower_means():
    assert narrower(torch.float16, torch.float32) is True
    assert narrower(torch.bfloat16, torch.float32) is True
    # Same width, three fewer mantissa bits: still a loss.
    assert narrower(torch.bfloat16, torch.float16) is True
    assert narrower(torch.float32, torch.float16) is False
    assert narrower(torch.float32, torch.float32) is False
    assert narrower(None, torch.float32) is False


# -------------------------------------------------------------- the real load
def test_an_fp32_checkpoint_into_fp16_parameters_raises():
    """The OPT case: torch itself performs this load without complaining."""
    source = _model(torch.float32)
    target = _model(torch.float16)
    sd = source.state_dict()

    # First, the thing that makes this worth a check: torch does it silently.
    target.load_state_dict(sd)
    assert target[0].weight.dtype == torch.float16

    with pytest.raises(PrecisionError, match="loses precision"):
        check_load_precision(_model(torch.float16), sd)

    info = check_load_precision(_model(torch.float16), sd, strict=False)
    assert info["narrowing"] is True
    assert info["checkpoint_dtype"] == "torch.float32"
    assert info["param_dtype"] == "torch.float16"


def test_a_widening_load_passes():
    sd = _model(torch.float16).state_dict()
    info = check_load_precision(_model(torch.float32), sd)
    assert info["narrowing"] is False


def test_an_identical_load_passes():
    sd = _model(torch.float32).state_dict()
    info = check_load_precision(_model(torch.float32), sd)
    assert info["narrowing"] is False
    assert info["half_precision_weights"] is False


def test_bfloat16_into_float16_is_caught_too():
    sd = _model(torch.float16).state_dict()
    with pytest.raises(PrecisionError):
        check_load_precision(_model(torch.bfloat16), sd)


def test_non_floating_entries_are_ignored():
    """An integer buffer in the checkpoint must not decide the dtype."""
    sd = _model(torch.float32).state_dict()
    sd["step"] = torch.tensor(7)
    info = check_load_precision(_model(torch.float32), sd)
    assert info["checkpoint_dtype"] == "torch.float32"
    assert info["narrowing"] is False


# ------------------------------------------------------------- the config case
def test_a_config_narrower_than_its_parameters_is_flagged():
    m = _model(torch.float32, config_dtype="float16")
    info = check_load_precision(m, strict=False)
    assert info["config_dtype"] == "torch.float16"
    assert info["config_narrower_than_params"] is True
    assert info["narrowing"] is True
    with pytest.raises(PrecisionError, match="save and reload"):
        check_load_precision(m)


def test_a_config_that_agrees_is_quiet():
    info = check_load_precision(_model(torch.float32, config_dtype="float32"))
    assert info["narrowing"] is False
    assert info["config_dtype"] == "torch.float32"


def test_a_torch_dtype_object_in_the_config_is_understood():
    info = model_precision(_model(torch.float32, config_dtype=torch.bfloat16))
    assert info["config_dtype"] == "torch.bfloat16"
    assert info["config_narrower_than_params"] is True


def test_half_precision_weights_warn_without_raising(caplog):
    with caplog.at_level(logging.WARNING):
        info = check_load_precision(_model(torch.float16))
    assert info["narrowing"] is False
    assert info["half_precision_weights"] is True
    assert any("not comparable" in r.message for r in caplog.records)


def test_mixed_parameter_dtypes_are_reported(caplog):
    m = _model(torch.float32)
    m[1].to(torch.float16)
    with caplog.at_level(logging.WARNING):
        info = check_load_precision(m, strict=False)
    assert info["mixed_param_dtypes"] == ["torch.float16", "torch.float32"]
    assert any("more than one floating dtype" in r.message
               for r in caplog.records)
    # The narrowest wins the tie, because it bounds what the model represents.
    assert info["param_dtype"] == "torch.float16"


# ----------------------------------------------------------- in the pipeline
def test_the_pipeline_records_the_precision_it_measured_at(tiny_model):
    from sal.pipeline import CompressionPipeline

    data = [{"input_ids": torch.randint(0, 100, (4, 16)),
             "labels": torch.randint(0, 100, (4, 16))} for _ in range(2)]
    pipe = CompressionPipeline(tiny_model, data, metric="loss", batch_size=4)
    assert pipe.precision["param_dtype"] == "torch.float32"
    report = pipe.validate()
    assert report.precision["param_dtype"] == "torch.float32"
    assert "torch.float32 weights" in report.stages[0].detail
    assert report.to_dict()["precision"]["narrowing"] is False


def test_the_pipeline_warns_on_a_half_precision_model(tiny_model, caplog):
    from sal.pipeline import CompressionPipeline

    tiny_model.half()
    data = [{"input_ids": torch.randint(0, 100, (4, 16)),
             "labels": torch.randint(0, 100, (4, 16))} for _ in range(2)]
    with caplog.at_level(logging.WARNING):
        pipe = CompressionPipeline(tiny_model, data, metric="loss",
                                   batch_size=4)
    assert pipe.precision["half_precision_weights"] is True
    assert any("not comparable" in r.message for r in caplog.records)
