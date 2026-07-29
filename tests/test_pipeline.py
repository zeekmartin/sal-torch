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


# ------------------------------------------------------- selection strategies
def test_selection_strategies_match_total_count(pipe):
    counts = {}
    for strategy in ("magnitude", "random"):
        heads = pipe.select_heads(0.25, strategy=strategy)
        counts[strategy] = len(heads)
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        assert {len(v) for v in by_layer.values()} == {2}   # uniform
    assert counts["magnitude"] == counts["random"] == 8


def test_random_strategy_is_seeded(tiny_model):
    a = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4, seed=7)
    b = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4, seed=7)
    c = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4, seed=8)
    assert a.select_heads(0.25, "random") == b.select_heads(0.25, "random")
    assert a.select_heads(0.25, "random") != c.select_heads(0.25, "random")


def test_magnitude_picks_the_weakest_heads(pipe):
    norms = pipe._head_norms()
    chosen = set(pipe.select_heads(0.25, "magnitude"))
    for li in range(4):
        picked = sorted(n for (l, h), n in norms.items() if l == li and (l, h) in chosen)
        kept = sorted(n for (l, h), n in norms.items() if l == li and (l, h) not in chosen)
        assert max(picked) <= min(kept)


def test_fi_guided_respects_layer_classes(pipe):
    """Never CRITICAL, never empties a layer — or refuses if nowhere is cheap."""
    from sal.fi import classify_layers, extract_activation_graph
    adj = extract_activation_graph(pipe.model, pipe.probe_dataset, num_samples=40,
                                   batch_size=4)
    layer_map = classify_layers(pipe.model, adj, num_heads_per_layer=8)
    classes = {getattr(c, "value", str(c)) for c in layer_map.values()}

    if classes == {"CRITICAL"}:
        # The tiny fixture lands here: no layer is safe, so refusing is correct.
        with pytest.raises(PipelineError, match="no IMMUNE or BUFFER"):
            pipe.select_heads(0.25, "fi_guided")
        return

    critical = {li for li, c in layer_map.items()
                if getattr(c, "value", str(c)) == "CRITICAL"}
    heads = pipe.select_heads(0.25, "fi_guided")
    assert not ({l for l, _ in heads} & critical)
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    assert all(len(v) <= 7 for v in by_layer.values())


def test_unknown_strategy_rejected(pipe):
    with pytest.raises(PipelineError, match="strategy must be one of"):
        pipe.select_heads(0.25, strategy="entropy")


def test_masked_compress_uses_the_selected_heads(tiny_model):
    """The masked path must mask what was selected, not a fresh random set."""
    pipe = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4)
    pipe.compress(pruning=0.25, quantization=None, slice_heads=False, strategy="magnitude")
    expected = set(CompressionPipeline(
        tiny_model, _data(), metric="loss", batch_size=4).select_heads(0.25, "magnitude"))
    assert set(pipe._pruned_heads) == expected
    assert "magnitude" in pipe.validate().last.detail


def test_fi_guided_cannot_be_sliced(tiny_model):
    """Non-uniform selection must be refused by slicing, not silently reshaped."""
    from sal.slicing import SlicingError
    pipe = CompressionPipeline(tiny_model, _data(), metric="loss", batch_size=4)
    with pytest.raises((SlicingError, PipelineError)):
        pipe.compress(pruning=0.25, quantization=None, slice_heads=True,
                      strategy="fi_guided")
