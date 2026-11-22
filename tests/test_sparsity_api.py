"""The v0.7.1 pruning API: counted zeros, and claims that match the hardware.

`prune_unstructured` and `prune_2of4` promote research code with a
layer-level equivalence test and 60 GPU-hours of use behind it into the
package. The tests that matter here are not that the algorithms work --
`test_exp33_sparse_equivalence.py` already pins them against the upstream
implementations -- but that the product surface cannot report something it
did not do:

* the zeros are **counted**, never inferred from the ratio that was asked
  for;
* 2:4 holds **in every group of four**, not on average, because a kernel
  reads groups and an average is no use to it;
* the record says `none by construction` for unstructured and
  `UNMEASURABLE` for 2:4, which are different claims and must not blur. This
  project spent a release withdrawing a "free" framing that came from
  exactly that blur.
"""
import pytest
import torch

import sal
from sal.sparsity import (SparsityError, attention_linear_layers,
                          linear_layers, prune_2of4, prune_unstructured,
                          sparsity_report)


def _calib(n=2, bs=2, sl=16):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(0, 100, (bs, sl))} for _ in range(n)]


def _zeros_in(layers) -> tuple:
    zeros = total = 0
    for mod in layers.values():
        w = mod.weight.data
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros, total


# ------------------------------------------------------------- the exports
def test_the_api_is_exported_from_the_package():
    for name in ("prune_unstructured", "prune_2of4", "SparsityError",
                 "sparsity_report"):
        assert hasattr(sal, name), name
        assert name in sal.__all__, name


def test_the_experiment_module_is_a_re_export_and_not_a_copy():
    """One implementation, or the published figures describe another module.

    exp33 produced 48 unstructured runs and 12 at 2:4 over three seeds
    against the code that used to live in `experiments/`. If the promotion
    had left a copy behind, those figures would describe a module the package
    no longer runs.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root / "experiments") not in sys.path:
        sys.path.insert(0, str(root / "experiments"))
    import exp33_sparse_core as core
    from sal import sparsity
    assert core.WandaPruner is sparsity.WandaPruner
    assert core.SparseGPTPruner is sparsity.SparseGPTPruner
    assert core.prune_model is sparsity.prune_model


# -------------------------------------------------------- counted, not told
@pytest.mark.parametrize("method", ["wanda", "sparsegpt"])
def test_unstructured_zeros_are_counted_and_match_the_ratio(tiny_model,
                                                            method):
    layers = attention_linear_layers(tiny_model)
    assert layers, "the fixture has no attention projections to prune"
    rec = prune_unstructured(tiny_model, _calib(), ratio=0.5, method=method,
                             scope="attention", log=lambda *a, **k: None)
    zeros, total = _zeros_in(layers)
    assert zeros / total == pytest.approx(0.5, abs=0.02), (zeros, total)
    assert rec["structure"] == "unstructured"
    assert rec["api"] == "sal.prune_unstructured"


def test_unstructured_promises_no_speedup(tiny_model):
    rec = prune_unstructured(tiny_model, _calib(), ratio=0.25,
                             log=lambda *a, **k: None)
    assert "none by construction" in rec["speedup"]
    assert "masked, not" in rec["speedup"]


def test_the_shapes_do_not_change(tiny_model):
    """Masked, not removed: a dense kernel does the same arithmetic."""
    before = {n: tuple(m.weight.shape)
              for n, m in attention_linear_layers(tiny_model).items()}
    params_before = sum(p.numel() for p in tiny_model.parameters())
    prune_unstructured(tiny_model, _calib(), ratio=0.5,
                       log=lambda *a, **k: None)
    after = {n: tuple(m.weight.shape)
             for n, m in attention_linear_layers(tiny_model).items()}
    assert before == after
    assert sum(p.numel() for p in tiny_model.parameters()) == params_before


# ------------------------------------------------------------- 2:4 per group
def test_2of4_holds_in_every_group_of_four(tiny_model):
    """Per group, not on average. A kernel reads groups.

    An average of one half over a row is satisfied by a row with eight zeros
    in one group of four and none in the next, which no sparse tensor core
    can use. This counts the groups.
    """
    rec = prune_2of4(tiny_model, _calib(), scope="attention",
                     log=lambda *a, **k: None)
    layers = attention_linear_layers(tiny_model)
    groups_checked = 0
    for name, mod in layers.items():
        w = mod.weight.data
        rows, cols = w.shape
        assert cols % 4 == 0, (name, cols)
        g = (w != 0).reshape(rows, cols // 4, 4).sum(-1)
        assert int(g.min()) == 2 and int(g.max()) == 2, (
            name, int(g.min()), int(g.max()))
        groups_checked += g.numel()
    assert groups_checked > 0
    assert rec["sparsity_fixed_by_format"] == 0.5


def test_2of4_calls_its_speedup_unmeasurable_and_not_none(tiny_model):
    """Two different claims, and this project has conflated them before."""
    rec = prune_2of4(tiny_model, _calib(), log=lambda *a, **k: None)
    assert "UNMEASURABLE" in rec["speedup"]
    assert "not none" in rec["speedup"]
    assert rec["structure"] == "2:4 semi-structured"


def test_2of4_takes_no_ratio(tiny_model):
    """The format fixes the sparsity; a ratio argument would imply otherwise."""
    with pytest.raises(TypeError):
        prune_2of4(tiny_model, _calib(), ratio=0.3)


# ------------------------------------------------------------- the refusals
def test_a_ratio_outside_the_open_unit_interval_is_refused(tiny_model):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(SparsityError, match="ratio"):
            prune_unstructured(tiny_model, _calib(), ratio=bad)


def test_an_unknown_scope_is_refused(tiny_model):
    with pytest.raises(SparsityError, match="scope"):
        prune_unstructured(tiny_model, _calib(), scope="everything")


def test_a_model_with_nothing_to_prune_is_refused():
    """Reported, not returned as a report of zeros."""
    import torch.nn as nn
    with pytest.raises(SparsityError, match="no attention Linear"):
        prune_unstructured(nn.Sequential(nn.ReLU()), _calib())


# --------------------------------------------------- the recovery handshake
def test_the_record_names_the_recovery_it_must_be_paired_with(tiny_model):
    rec = prune_unstructured(tiny_model, _calib(), ratio=0.25,
                             log=lambda *a, **k: None)
    assert "preserve_sparsity=True" in rec["pair_with"]


def test_recovery_preserves_the_pattern_this_api_produced(tiny_model):
    """The handshake, executed: prune, recover, and count the drift.

    `preserve_sparsity=True` is already the default, so this checks that the
    pattern THIS api writes is the pattern that stage defends -- the two
    were developed separately and nothing before this ran them together.
    """
    from sal.recovery import recover_model

    prune_unstructured(tiny_model, _calib(), ratio=0.5,
                       log=lambda *a, **k: None)
    layers = attention_linear_layers(tiny_model)
    zeros_before, total = _zeros_in(layers)
    assert zeros_before > 0

    data = [{"input_ids": torch.randint(0, 100, (4, 16)),
             "labels": torch.randint(0, 100, (4, 16))} for _ in range(4)]
    rec = recover_model(tiny_model, data, epochs=1, lr=1e-3, batch_size=4,
                        domain="same as eval", preserve_sparsity=True)
    zeros_after, _ = _zeros_in(layers)
    assert zeros_after == zeros_before, (zeros_before, zeros_after)
    assert rec["sparsity_drift"] == 0.0
    assert rec["mask_preserved"] is True


def test_the_sparsity_report_counts_what_is_there(tiny_model):
    layers = linear_layers(tiny_model)
    prune_unstructured(tiny_model, _calib(), ratio=0.5, scope="whole",
                       log=lambda *a, **k: None)
    rep = sparsity_report(tiny_model, layers)
    zeros, total = _zeros_in(layers)
    assert rep["target_zeros"] == zeros
    assert rep["target_params"] == total
