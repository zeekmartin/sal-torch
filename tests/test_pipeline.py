"""Tests for CompressionPipeline (CPU, tiny model fixture)."""
import json

import pytest
import torch
import torch.nn as nn

from sal.pipeline import (CompressionPipeline, PipelineError, SMALL_MODEL_PARAMS,
                          detect_adapters)


def _data(n_batches=4, bs=4, sl=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, 100, (bs, sl), generator=g),
             "labels": torch.randint(0, 100, (bs, sl), generator=g)}
            for _ in range(n_batches)]


@pytest.fixture
def pipe(tiny_model):
    return CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4,
                               model_name="tiny")


def test_pipeline_scan(pipe):
    scan = pipe.scan()
    assert scan.num_layers == 4
    assert scan.num_heads == 8
    assert 0.0 <= scan.fi_score <= 1.0
    assert scan.total_params > 0
    assert scan.quantization["int4_size_mb"] < scan.quantization["original_size_mb"]
    rec = scan.recommendation
    assert "layers" in rec and "fragility index" in rec
    # Below the size threshold the recommendation must say so rather than sell SAL.
    assert "marginal" in rec


def test_pipeline_select_heads_is_uniform(pipe):
    heads = pipe.select_heads(0.25)
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    assert sorted(by_layer) == [0, 1, 2, 3]
    assert {len(v) for v in by_layer.values()} == {2}      # 25% of 8 heads


def test_pipeline_select_heads_rejects_degenerate(pipe):
    with pytest.raises(PipelineError, match="removes no heads"):
        pipe.select_heads(0.01)
    with pytest.raises(PipelineError, match="every head"):
        pipe.select_heads(1.0)


def test_pipeline_compress_produces_smaller_model(pipe):
    pipe.compress(pruning=0.25, quantization=None, slice_heads=True)
    report = pipe.validate()
    assert len(report.stages) == 2
    assert report.last.size_mb < report.first.size_mb
    assert report.size_ratio > 1.0
    assert pipe.model.config.num_attention_heads == 6


def test_pipeline_compress_masked_keeps_size(pipe):
    """Masking measures what pruning costs; it does not make anything smaller."""
    pipe.compress(pruning=0.25, quantization=None, slice_heads=False)
    report = pipe.validate()
    assert report.last.size_mb == pytest.approx(report.first.size_mb)
    assert "not removed" in report.last.detail


def test_pipeline_quantize_stage(pipe):
    pipe.compress(pruning=None, quantization="int8")
    report = pipe.validate()
    assert len(report.stages) == 2
    assert report.last.size_mb < report.first.size_mb
    assert "quantized" in report.last.name


def test_full_pipeline(tiny_model, tmp_path):
    pipe = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4)
    pipe.scan()
    pipe.sal_train(_data(seed=1), epochs=1, prune_fraction=0.25, lr=1e-3)
    pipe.compress(pruning=0.25, quantization="int8", slice_heads=True)
    report = pipe.validate()

    names = [s.name for s in report.stages]
    assert names == ["original", "sal_trained", "pruned+sliced", "quantized (int8)"]
    assert report.size_ratio > 1.0
    assert all(s.accuracy is not None for s in report.stages)
    assert "MB ->" in report.summary
    assert "stage" in report.table

    out = tmp_path / "report.json"
    report.save(str(out))
    data = json.loads(out.read_text())
    assert len(data["stages"]) == 4
    assert data["scan"] is not None


def test_pipeline_export(tiny_model, tmp_path):
    pipe = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4)
    pipe.compress(pruning=0.25, quantization=None, slice_heads=True)
    result = pipe.export(str(tmp_path / "compressed"))
    assert result["format"] == "torch.save"
    assert result["roundtrip_verified"] is True

    reloaded = torch.load(tmp_path / "compressed" / "model.pt", weights_only=False)
    batch = {"input_ids": torch.randint(0, 100, (2, 16))}
    with torch.no_grad():
        a = pipe.model(**batch).logits
        b = reloaded(**batch).logits
    assert torch.allclose(a, b)
    # Standalone: no hooks anywhere in the exported model.
    for block in reloaded.transformer.h:
        assert not block.attn.out_proj._forward_pre_hooks


def test_pipeline_refuses_lora(tiny_model):
    """A LoRA-wrapped model must be refused with an actionable message."""
    class FakeLora(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.lora_A = nn.Linear(64, 8, bias=False)
            self.lora_B = nn.Linear(8, 64, bias=False)

        def forward(self, x):
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    tiny_model.transformer.h[0].attn.q_proj = FakeLora(
        tiny_model.transformer.h[0].attn.q_proj)
    assert detect_adapters(tiny_model) is not None
    with pytest.raises(PipelineError, match="full fine-tuning"):
        CompressionPipeline(tiny_model, _data())


def test_pipeline_refuses_peft_config_attribute(tiny_model):
    tiny_model.peft_config = {"default": object()}
    with pytest.raises(PipelineError, match="adapters"):
        CompressionPipeline(tiny_model, _data())


def test_detect_adapters_false_on_plain_model(tiny_model):
    assert detect_adapters(tiny_model) is None


def test_small_model_warning(tiny_model, caplog):
    with caplog.at_level("WARNING"):
        CompressionPipeline(tiny_model, _data())
    assert any("marginal" in r.message for r in caplog.records)
    assert SMALL_MODEL_PARAMS == 100_000_000


def test_accuracy_floor_stops_the_run(tiny_model):
    """A floor must stop the pipeline rather than return a small broken model."""
    # metric="loss": the floor is an upper bound, so an impossible one trips at once.
    pipe = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4,
                               accuracy_floor=0.0001)
    with pytest.raises(PipelineError, match="accuracy_floor"):
        pipe.compress(pruning=0.25, quantization=None)


def test_pipeline_report(tiny_model, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("fpdf")
    pipe = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4,
                               model_name="tiny")
    pipe.scan()
    pipe.compress(pruning=0.25, quantization="int8", slice_heads=True)
    out = tmp_path / "compression.pdf"
    pipe.report().save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_invalid_metric(tiny_model):
    with pytest.raises(ValueError, match="metric"):
        CompressionPipeline(tiny_model, _data(), metric="f1")
