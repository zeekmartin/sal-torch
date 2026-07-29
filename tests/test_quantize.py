"""Tests for the quantization wrapper (CPU, tiny model fixture)."""
import pytest
import torch
import torch.nn as nn

from sal.quantize import (DEFAULT_SKIP, QuantizationError, available_backends,
                          has_bitsandbytes, model_size_mb, quantize, quantize_info)

HAS_BNB = has_bitsandbytes()
HAS_CUDA = torch.cuda.is_available()


def _batch(bs=2, sl=16):
    torch.manual_seed(0)
    return {"input_ids": torch.randint(0, 100, (bs, sl))}


def test_int8_wrapper(tiny_model):
    before = model_size_mb(tiny_model)
    q = quantize(tiny_model, method="int8")
    after = model_size_mb(q)
    assert after < before
    assert q._sal_quantization["method"] == "int8"
    assert q._sal_quantization["layers_quantized"] > 0
    with torch.no_grad():
        out = q(**_batch())
    assert out.logits.shape == (2, 16, 100)
    assert torch.isfinite(out.logits).all()


def test_int8_leaves_original_untouched(tiny_model):
    quantize(tiny_model, method="int8")
    assert isinstance(tiny_model.transformer.h[0].attn.q_proj, nn.Linear)
    assert not hasattr(tiny_model, "_sal_quantization")


def test_output_head_is_not_quantized(tiny_model):
    """Quantizing the vocabulary projection costs more accuracy than it saves."""
    q = quantize(tiny_model, method="int8")
    assert isinstance(q.head, nn.Linear)
    assert type(q.head).__name__ == "Linear"


@pytest.mark.skipif(not (HAS_BNB and HAS_CUDA), reason="needs bitsandbytes + CUDA")
def test_int4_wrapper(tiny_model):
    before = model_size_mb(tiny_model)
    q = quantize(tiny_model, method="int4")
    assert model_size_mb(q) < before
    assert q._sal_quantization["backend"] == "bitsandbytes-nf4"


def test_int4_unavailable_raises_clearly(tiny_model):
    """Without a CUDA bitsandbytes there is no int4 path; say so, don't fake it."""
    if HAS_BNB and HAS_CUDA:
        pytest.skip("bitsandbytes int4 is available here")
    with pytest.raises(QuantizationError, match="No backend available"):
        quantize(tiny_model, method="int4")


def test_auto_backend_selects_available(tiny_model):
    q = quantize(tiny_model, method="int8", backend="auto")
    expected = "bitsandbytes-int8" if (HAS_BNB and HAS_CUDA) else "torch-dynamic-int8"
    assert q._sal_quantization["backend"] == expected


def test_explicit_backend_error_names_the_backend(tiny_model):
    with pytest.raises(QuantizationError, match="torch_ao"):
        quantize(tiny_model, method="int4", backend="torch_ao")


def test_invalid_arguments(tiny_model):
    with pytest.raises(ValueError, match="method"):
        quantize(tiny_model, method="int2")
    with pytest.raises(ValueError, match="backend"):
        quantize(tiny_model, method="int8", backend="gptq")


def test_available_backends():
    b8 = available_backends("int8")
    assert "torch_ao" in b8
    b4 = available_backends("int4")
    assert "torch_ao" not in b4          # no int4 path in torch.ao
    if HAS_BNB and HAS_CUDA:
        assert "bitsandbytes" in b8 and "bitsandbytes" in b4


def test_quantize_info(tiny_model):
    info = quantize_info(tiny_model)
    assert info["original_size_mb"] > 0
    assert info["int8_size_mb"] < info["original_size_mb"]
    assert info["int4_size_mb"] < info["int8_size_mb"]
    assert "backends_available" in info
    assert isinstance(info["int4_available"], bool)


def test_quantize_info_excludes_the_head(tiny_model):
    """The skipped output head must not be counted as quantizable savings."""
    info = quantize_info(tiny_model)
    head_mb = tiny_model.head.weight.numel() * tiny_model.head.weight.element_size() / 1e6
    assert info["quantizable_mb"] < info["original_size_mb"] - head_mb + 1e-6


def test_skip_list_is_respected(tiny_model):
    q = quantize(tiny_model, method="int8", skip=DEFAULT_SKIP + ("q_proj",))
    assert isinstance(q.transformer.h[0].attn.q_proj, nn.Linear)
    assert type(q.transformer.h[0].attn.q_proj).__name__ == "Linear"
    # Something else still got quantized.
    assert q._sal_quantization["layers_quantized"] > 0
