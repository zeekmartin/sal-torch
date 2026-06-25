"""Architecture integration tests on real HuggingFace models.

Verifies SALConfig.auto() + HeadMasker on each supported architecture:
  1. load the model, 2. auto-detect layers/heads, 3. install + activate masker,
  4. one forward pass, 5. output shape correct AND differs from unmasked,
  6. clean removal.

These need network + model downloads (and ideally GPU). They are marked
``integration`` and skipped unless ``pytest --run-integration`` is passed.

The ``check_architecture`` helper is plain (no pytest fixtures) so the Modal
runner in scripts/modal_integration_test.py can call the exact same logic.
"""
from __future__ import annotations
import pytest
import torch

# (model_name, loader, input_kind, expected_layers, expected_heads)
ARCH_CASES = [
    ("distilbert-base-uncased", "AutoModelForSequenceClassification", "text", 6, 12),
    ("gpt2", "AutoModelForCausalLM", "text", 12, 12),
    ("google/vit-base-patch16-224", "AutoModelForImageClassification", "image", 12, 12),
    ("bert-base-uncased", "AutoModelForSequenceClassification", "text", 12, 12),
]


def _load_model(model_name: str, loader: str):
    import transformers
    cls = getattr(transformers, loader)
    return cls.from_pretrained(model_name)


def _make_inputs(model, input_kind: str):
    torch.manual_seed(0)
    if input_kind == "image":
        size = getattr(model.config, "image_size", 224)
        nch = getattr(model.config, "num_channels", 3)
        return {"pixel_values": torch.randn(2, nch, size, size)}
    vocab = getattr(model.config, "vocab_size", 30522)
    ids = torch.randint(0, vocab, (2, 16))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def check_architecture(model_name: str, loader: str, input_kind: str,
                       expected_layers: int, expected_heads: int, device: str = "cpu") -> dict:
    """Run the full arch check. Raises AssertionError on failure; returns a report."""
    from sal import SALConfig
    from sal.masker import HeadMasker

    model = _load_model(model_name, loader).to(device)
    model.eval()

    # 2. auto-detect
    config = SALConfig.auto(model, prune_fraction=0.33)
    assert config.num_layers == expected_layers, \
        f"{model_name}: detected {config.num_layers} layers, expected {expected_layers}"
    assert config.num_heads_per_layer == expected_heads, \
        f"{model_name}: detected {config.num_heads_per_layer} heads, expected {expected_heads}"

    inputs = {k: v.to(device) for k, v in _make_inputs(model, input_kind).items()}

    # baseline (unmasked) forward
    with torch.no_grad():
        clean = model(**inputs).logits.clone()

    # 3. install + activate
    masker = HeadMasker(model, config, seed=42)
    masker.install()
    assert len(masker._hooks) == expected_layers, \
        f"{model_name}: attached {len(masker._hooks)} hooks, expected {expected_layers}"
    masker.activate()
    pruned = config.total_heads - masker._active_head_count()
    assert pruned == config.num_heads_to_prune, \
        f"{model_name}: pruned {pruned}, expected {config.num_heads_to_prune}"

    # 4 + 5. forward, shape preserved, output changed
    with torch.no_grad():
        masked = model(**inputs).logits.clone()
    assert masked.shape == clean.shape, f"{model_name}: shape changed {clean.shape}->{masked.shape}"
    max_diff = (masked - clean).abs().max().item()
    assert not torch.allclose(clean, masked, atol=1e-6), \
        f"{model_name}: masking did not change the output (max_diff={max_diff:.2e})"

    # 6. clean removal — output returns to baseline
    masker.remove()
    assert len(masker._hooks) == 0
    with torch.no_grad():
        restored = model(**inputs).logits.clone()
    assert torch.allclose(clean, restored, atol=1e-5), \
        f"{model_name}: output not restored after masker.remove()"

    return {"model": model_name, "layers": config.num_layers, "heads": config.num_heads_per_layer,
            "hooks": expected_layers, "pruned_heads": pruned, "max_abs_diff": max_diff,
            "passed": True}


@pytest.mark.integration
@pytest.mark.parametrize("model_name,loader,input_kind,layers,heads", ARCH_CASES,
                         ids=[c[0] for c in ARCH_CASES])
def test_architecture(model_name, loader, input_kind, layers, heads):
    report = check_architecture(model_name, loader, input_kind, layers, heads)
    assert report["passed"]


def run_all(verbose: bool = True, device: str = "cpu") -> list[dict]:
    """Run every architecture case sequentially (used by the Modal runner)."""
    results = []
    for name, loader, kind, layers, heads in ARCH_CASES:
        try:
            r = check_architecture(name, loader, kind, layers, heads, device=device)
        except Exception as e:  # noqa: BLE001 — report, don't crash the whole sweep
            r = {"model": name, "passed": False, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if verbose:
            print(f"[arch] {r}", flush=True)
    return results
