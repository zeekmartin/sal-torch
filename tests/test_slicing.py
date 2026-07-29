"""Tests for head slicing (CPU, tiny model fixture)."""
import io

import pytest
import torch

from sal.slicing import (SlicingError, head_savings, slice_heads, verify_slicing)


def _batch(bs=2, sl=16):
    torch.manual_seed(0)
    return {"input_ids": torch.randint(0, 100, (bs, sl))}


def _uniform(layers=4, heads=(1, 5)):
    """Remove the same heads from every layer — the only shape slicing accepts."""
    return [(l, h) for l in range(layers) for h in heads]


def test_slice_removes_heads(tiny_model):
    attn = tiny_model.transformer.h[0].attn
    assert attn.q_proj.weight.shape == (64, 64)
    assert attn.out_proj.weight.shape == (64, 64)

    sliced = slice_heads(tiny_model, _uniform())
    a = sliced.transformer.h[0].attn
    # 8 heads -> 6, head_dim stays 8: Q/K/V lose rows, O loses columns.
    assert a.q_proj.weight.shape == (48, 64)
    assert a.k_proj.weight.shape == (48, 64)
    assert a.v_proj.weight.shape == (48, 64)
    assert a.out_proj.weight.shape == (64, 48)
    assert a.nh == 6
    assert a.hd == 8          # head_dim must not change


def test_sliced_output_matches_masked(tiny_model):
    batch = _batch()
    remove = _uniform()
    sliced = slice_heads(tiny_model, remove, verify_input=batch)
    # Bit-exact on a model this small: slicing and masking are the same arithmetic.
    assert verify_slicing(tiny_model, sliced, remove, batch) < 1e-5


def test_slicing_leaves_the_original_untouched(tiny_model):
    batch = _batch()
    with torch.no_grad():
        before = tiny_model(**batch).logits.clone()
    slice_heads(tiny_model, _uniform())
    assert tiny_model.config.num_attention_heads == 8
    assert tiny_model.transformer.h[0].attn.q_proj.weight.shape == (64, 64)
    with torch.no_grad():
        assert torch.allclose(tiny_model(**batch).logits, before)


def test_config_updated(tiny_model):
    assert tiny_model.config.num_attention_heads == 8
    sliced = slice_heads(tiny_model, _uniform())
    assert sliced.config.num_attention_heads == 6


def test_sliced_model_runs(tiny_model):
    sliced = slice_heads(tiny_model, _uniform())
    batch = _batch()
    with torch.no_grad():
        out = sliced(**batch)
    assert out.logits.shape == (2, 16, 100)
    assert torch.isfinite(out.logits).all()
    # The whole point: no hooks left anywhere.
    for block in sliced.transformer.h:
        assert not block.attn.out_proj._forward_pre_hooks
        assert not block.attn.out_proj._forward_hooks


def test_roundtrip_save_load(tiny_model, tmp_path):
    """A sliced model must survive serialization and stay standalone."""
    sliced = slice_heads(tiny_model, _uniform())
    batch = _batch()
    with torch.no_grad():
        expected = sliced(**batch).logits.clone()

    buf = io.BytesIO()
    torch.save(sliced, buf)
    buf.seek(0)
    reloaded = torch.load(buf, weights_only=False)

    with torch.no_grad():
        assert torch.allclose(reloaded(**batch).logits, expected)
    assert reloaded.config.num_attention_heads == 6
    assert reloaded.transformer.h[0].attn.q_proj.weight.shape == (48, 64)


def test_parameter_count_drops(tiny_model):
    before = sum(p.numel() for p in tiny_model.parameters())
    sliced = slice_heads(tiny_model, _uniform())
    after = sum(p.numel() for p in sliced.parameters())
    assert after < before
    # Prediction must match what actually happened.
    assert head_savings(tiny_model, 2)["removed_params"] == before - after


def test_non_uniform_removal_rejected(tiny_model):
    with pytest.raises(SlicingError, match="same number of heads"):
        slice_heads(tiny_model, [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)])


def test_out_of_range_rejected(tiny_model):
    with pytest.raises(SlicingError, match="Layer index out of range"):
        slice_heads(tiny_model, [(9, 1)])
    with pytest.raises(SlicingError, match="Head index out of range"):
        slice_heads(tiny_model, [(l, 99) for l in range(4)])


def test_removing_every_head_rejected(tiny_model):
    with pytest.raises(SlicingError, match="Cannot remove all"):
        slice_heads(tiny_model, [(l, h) for l in range(4) for h in range(8)])


def test_empty_removal_rejected(tiny_model):
    with pytest.raises(SlicingError, match="Nothing to remove"):
        slice_heads(tiny_model, [])


def test_grouped_query_attention_rejected(tiny_model):
    """GQA ties query heads to shared KV heads; slicing must refuse, not guess."""
    tiny_model.config.num_key_value_heads = 4      # 8 query heads over 4 KV heads
    try:
        with pytest.raises(SlicingError, match="grouped-query"):
            slice_heads(tiny_model, _uniform())
    finally:
        del tiny_model.config.num_key_value_heads


def test_slice_fraction_of_heads(tiny_model):
    """Removing a single head per layer works as well as several."""
    sliced = slice_heads(tiny_model, [(l, 0) for l in range(4)], verify_input=_batch())
    assert sliced.config.num_attention_heads == 7
    assert sliced.transformer.h[0].attn.q_proj.weight.shape == (56, 64)
