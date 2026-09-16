"""Self-supervised vision architectures: I-JEPA and DINOv2 (v0.5.1).

Not integration tests — these build tiny models from a local config, so no
network, no checkpoints, no GPU. What they pin is that ``detect_architecture``
returns a pattern that *actually resolves*, which is the failure mode a
registry entry invites: a plausible-looking string that finds nothing, after
which ``get_attention_modules`` silently falls back to a different pattern and
the registry entry is never exercised again.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from sal import arch_support                                        # noqa: E402
from sal.config import SALConfig                                    # noqa: E402
from sal.masker import HeadMasker                                   # noqa: E402

NL, NH, HS = 2, 4, 64
IMG, PATCH = 32, 16


def _ijepa():
    from transformers import IJepaConfig, IJepaModel
    return IJepaModel(IJepaConfig(hidden_size=HS, num_hidden_layers=NL,
                                  num_attention_heads=NH, intermediate_size=HS * 2,
                                  image_size=IMG, patch_size=PATCH))


def _dinov2():
    from transformers import Dinov2Config, Dinov2Model
    return Dinov2Model(Dinov2Config(hidden_size=HS, num_hidden_layers=NL,
                                    num_attention_heads=NH, intermediate_size=HS * 2,
                                    image_size=IMG, patch_size=PATCH))


MODELS = [("ijepa", _ijepa), ("dinov2", _dinov2)]


@pytest.mark.parametrize("model_type,build", MODELS, ids=[m[0] for m in MODELS])
def test_architecture_is_registered(model_type, build):
    info = arch_support.detect_architecture(build())
    assert info.model_type == model_type
    assert info.num_layers == NL
    assert info.num_heads == NH
    assert model_type in arch_support.supported_architectures()


@pytest.mark.parametrize("model_type,build", MODELS, ids=[m[0] for m in MODELS])
def test_registered_pattern_actually_resolves(model_type, build):
    """The registry's own pattern must find the layers — not a fallback's."""
    model = build()
    pattern = arch_support.detect_architecture(model).attention_pattern
    assert len(arch_support._find_by_pattern(model, pattern)) == NL


@pytest.mark.parametrize("model_type,build", MODELS, ids=[m[0] for m in MODELS])
def test_output_projections_found(model_type, build):
    model = build()
    projs = arch_support.get_output_projections(model)
    assert len(projs) == NL
    assert all(p.weight.shape == (HS, HS) for p in projs)


@pytest.mark.parametrize("model_type,build", MODELS, ids=[m[0] for m in MODELS])
def test_qkv_projections_found(model_type, build):
    """Head-level weight slicing needs Q/K/V, not just the output projection."""
    attn = arch_support.get_attention_modules(build())[0]
    qkv = arch_support.get_qkv_projections(attn)
    assert qkv is not None and qkv["mode"] == "separate"


@pytest.mark.parametrize("model_type,build", MODELS, ids=[m[0] for m in MODELS])
def test_masking_changes_the_output(model_type, build):
    """The whole point: masked heads must actually change what comes out."""
    torch.manual_seed(0)
    model = build().eval()
    pixels = torch.randn(2, 3, IMG, IMG)

    with torch.no_grad():
        clean = model(pixel_values=pixels).last_hidden_state

    cfg = SALConfig.auto(model, prune_fraction=0.5, prune_start_ratio=0.0)
    masker = HeadMasker(model, cfg, seed=0)
    masker.install()
    try:
        masker.activate()
        assert masker.stats["pruned_heads"] == NL * NH // 2
        with torch.no_grad():
            masked = model(pixel_values=pixels).last_hidden_state
    finally:
        masker.remove()

    assert masked.shape == clean.shape
    assert not torch.allclose(masked, clean), "masking had no effect on the output"

    with torch.no_grad():                       # hooks gone, original behaviour back
        assert torch.allclose(model(pixel_values=pixels).last_hidden_state, clean)


@pytest.mark.parametrize("model_type,build", MODELS, ids=[m[0] for m in MODELS])
def test_fi_scan_runs(model_type, build):
    """FI hooks the same projections the masker does, on a vision batch."""
    from sal.scanner import FIScanner
    probe = [{"pixel_values": torch.randn(4, 3, IMG, IMG)} for _ in range(3)]
    res = FIScanner(build(), probe, num_samples=8).scan()
    assert 0.0 <= res.fi_score <= 1.0
    assert res.num_layers == NL
    assert res.num_heads_per_layer == NH


def test_detect_architecture_verifies_its_pattern(monkeypatch):
    """A registry pattern that finds nothing must be replaced, not reported.

    transformers 4.x lays I-JEPA out as ``encoder.layer.{}.attention`` while 5.x
    uses ``layers.{}.attention``. Whichever spelling the registry holds, the
    pattern handed back has to resolve against the model in front of it —
    ``get_attention_modules`` would otherwise fall back silently and everything
    would keep working while ArchInfo advertised a pattern matching zero modules.
    """
    model = _ijepa()
    monkeypatch.setitem(arch_support._REGISTRY, "ijepa", "nonsense.{}.path")

    info = arch_support.detect_architecture(model)
    assert info.attention_pattern != "nonsense.{}.path"
    assert len(arch_support._find_by_pattern(model, info.attention_pattern)) == NL


def test_detect_architecture_keeps_a_working_pattern(monkeypatch):
    """When the registry is right, it is returned untouched."""
    model = _ijepa()
    info = arch_support.detect_architecture(model)
    assert len(arch_support._find_by_pattern(model, info.attention_pattern)) == NL
