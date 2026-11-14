"""OPT support in arch_support, pinned.

OPT is the first family in the registry whose transformer stack is nested one
level below ``base_model``: ``OPTModel``'s only child is ``decoder``, so the
attention container is ``decoder.layers.{i}.self_attn`` and not the
``layers.{i}.self_attn`` that llama, mistral, phi, gemma and qwen use. A
pattern written without that prefix resolves to nothing and the masker
silently finds no heads, so this is worth a test even though it is one line
of data.

The unit tests here need no model. The real-model check is marked
``integration`` like the rest of ``tests/test_arch_integration.py``.
"""
from __future__ import annotations

import pytest
import torch

from sal import arch_support


def test_opt_is_registered_with_the_decoder_nested_path():
    assert arch_support._REGISTRY["opt"] == "decoder.layers.{}.self_attn"


def test_the_decoder_nested_pattern_is_also_a_fallback():
    """So a decoder-nested family we have not registered still resolves."""
    assert "decoder.layers.{}.self_attn" in arch_support._FALLBACK_PATTERNS


def test_opt_is_not_confused_with_the_flat_llama_layout():
    """The bug this guards: OPT silently matching nothing, or llama's path."""
    assert arch_support._REGISTRY["opt"] != arch_support._REGISTRY["llama"]
    assert arch_support._REGISTRY["opt"].startswith("decoder.")


def test_opt_resolves_against_a_decoder_nested_stub():
    """A stub with OPT's nesting, so the resolver is exercised without a
    download: base_model -> decoder -> layers[i] -> self_attn -> out_proj."""

    class Attn(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.out_proj = torch.nn.Linear(h, h)

    class Layer(torch.nn.Module):
        def __init__(self, h, ffn):
            super().__init__()
            self.self_attn = Attn(h)
            self.fc1 = torch.nn.Linear(h, ffn)
            self.fc2 = torch.nn.Linear(ffn, h)

    class Decoder(torch.nn.Module):
        def __init__(self, n, h, ffn):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [Layer(h, ffn) for _ in range(n)])

    class Stub(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self, n=4, h=16, ffn=64):
            super().__init__()

            class Inner(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.decoder = Decoder(n, h, ffn)

            self.model = Inner()

        @property
        def base_model(self):
            return self.model

    stub = Stub()
    mods = arch_support.get_attention_modules(
        stub, arch_support._REGISTRY["opt"])
    assert len(mods) == 4, mods
    assert all(hasattr(m, "out_proj") for m in mods)
    assert all(arch_support.get_output_projection(m) is not None
               for m in mods)

    # and it resolves with no pattern at all, through the new fallback
    assert len(arch_support.get_attention_modules(stub)) == 4

    # and the FFN expansion projection is findable by the out>in rule
    layer0 = stub.base_model.decoder.layers[0]
    expansion = [n for n, m in layer0.named_children()
                 if isinstance(m, torch.nn.Linear)
                 and m.out_features > m.in_features]
    assert expansion == ["fc1"], expansion


@pytest.mark.integration
def test_opt_125m_detects_twelve_by_twelve():
    """Needs a download; run with --run-integration."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    info = arch_support.detect_architecture(model)
    assert info.num_layers == 12
    assert info.num_heads == 12
    assert info.hidden_size == 768
    projs = arch_support.get_output_projections(
        model, info.attention_pattern)
    assert len(projs) == info.num_layers, len(projs)
