# -*- coding: utf-8 -*-
"""Grouping bounds the memory of the pruning pass and changes nothing else.

``prune_model`` keeps one pruner per target alive through a single
calibration pass. Each SparseGPT pruner holds an ``in_features**2`` fp32
Hessian, so OPT-6.7B needs 42.0 GiB of them against 7.9 at 1.3B, and with
fp32 weights that exceeds a 64 GiB GCD -- the OOM the 6.7B reference gate
hit right after reproducing the dense perplexity to four decimals.

``group_size`` prunes the targets in groups, one calibration pass each.
The semantics must be untouched: every published figure in this campaign
was measured with calibration activations taken from the DENSE model, and
the sequential reference implementations differ on exactly that point. So
a group's pruned weights are buffered and the originals restored before
the next pass.

These tests are the proof of that claim, not an illustration of it: they
require the masks and the weights to be **bit-identical**, not close.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp33_sparse_core import prune_model  # noqa: E402

SPARSITY = 0.5
NCALIB = 6


class Tiny(nn.Module):
    """Six linear layers in sequence, enough for several groups."""

    def __init__(self, width=16, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = torch.relu(b(x))
        return x


def _fresh(seed=0):
    torch.manual_seed(seed)
    m = Tiny()
    layers = {"blocks.%d" % i: b for i, b in enumerate(m.blocks)}
    calib = [torch.randn(4, 16) for _ in range(NCALIB)]
    return m, layers, calib


def _state(layers):
    return {n: lyr.weight.data.detach().clone() for n, lyr in layers.items()}


@pytest.mark.parametrize("method", ["sparsegpt", "wanda"])
@pytest.mark.parametrize("group_size", [1, 2, 3, 6])
def test_grouped_equals_whole_model(method, group_size):
    """Bit-identical weights, for every group size and both pruners."""
    m1, l1, c1 = _fresh()
    prune_model(m1, l1, c1, method, SPARSITY, log=lambda *a, **k: None)
    whole = _state(l1)

    m2, l2, c2 = _fresh()
    prune_model(m2, l2, c2, method, SPARSITY, log=lambda *a, **k: None,
                group_size=group_size)
    grouped = _state(l2)

    assert set(whole) == set(grouped)
    for n in whole:
        assert torch.equal(whole[n], grouped[n]), (
            "%s differs at group_size=%d: max |delta| %g"
            % (n, group_size,
               (whole[n] - grouped[n]).abs().max().item()))


@pytest.mark.parametrize("method", ["sparsegpt", "wanda"])
def test_grouped_masks_are_identical(method):
    """The zero pattern, stated separately from the values."""
    m1, l1, c1 = _fresh()
    prune_model(m1, l1, c1, method, SPARSITY, log=lambda *a, **k: None)
    m2, l2, c2 = _fresh()
    prune_model(m2, l2, c2, method, SPARSITY, log=lambda *a, **k: None,
                group_size=2)
    for n in l1:
        a = l1[n].weight.data == 0
        b = l2[n].weight.data == 0
        assert torch.equal(a, b), "%s: masks differ" % n


def test_grouping_actually_prunes():
    """A check that passes on two untouched models proves nothing."""
    m, layers, calib = _fresh()
    before = _state(layers)
    prune_model(m, layers, calib, "wanda", SPARSITY,
                log=lambda *a, **k: None, group_size=2)
    for n, lyr in layers.items():
        frac = float((lyr.weight.data == 0).float().mean())
        assert abs(frac - SPARSITY) < 0.1, "%s: zero fraction %.3f" % (n,
                                                                       frac)
        assert not torch.equal(before[n], lyr.weight.data)


def test_every_pass_sees_the_dense_model():
    """The property grouping must preserve, tested directly.

    If a later group saw an earlier group's pruning, its Hessian would be
    built from different activations and its result would differ from the
    single-pass one. Pruning the LAST group alone must therefore give the
    same weights whether or not the earlier groups were pruned first.
    """
    m1, l1, c1 = _fresh()
    last = {"blocks.5": l1["blocks.5"]}
    prune_model(m1, last, c1, "sparsegpt", SPARSITY,
                log=lambda *a, **k: None)
    alone = l1["blocks.5"].weight.data.detach().clone()

    m2, l2, c2 = _fresh()
    prune_model(m2, l2, c2, "sparsegpt", SPARSITY,
                log=lambda *a, **k: None, group_size=2)
    in_groups = l2["blocks.5"].weight.data.detach().clone()

    assert torch.equal(alone, in_groups), (
        "the last layer's result depends on whether earlier layers were "
        "pruned first, so a pass is not seeing the dense model")


def test_group_size_larger_than_the_model_is_one_group():
    m1, l1, c1 = _fresh()
    prune_model(m1, l1, c1, "wanda", SPARSITY, log=lambda *a, **k: None)
    m2, l2, c2 = _fresh()
    prune_model(m2, l2, c2, "wanda", SPARSITY, log=lambda *a, **k: None,
                group_size=999)
    for n in l1:
        assert torch.equal(l1[n].weight.data, l2[n].weight.data)


@pytest.mark.parametrize("method", ["sparsegpt", "wanda"])
def test_n_m_sparsity_is_equivalent_too(method):
    """The 2:4 path, which the 6.7B gate exercises and the tests above did not.

    `prunen`/`prunem` travel through the grouped call like everything
    else, but "it is passed through" is an argument, not a measurement:
    n:m picks its zeros within each group of m consecutive weights, so a
    bug there would show as a different pattern rather than a crash.
    """
    m1, l1, c1 = _fresh()
    prune_model(m1, l1, c1, method, 0.5, log=lambda *a, **k: None,
                prunen=2, prunem=4)
    m2, l2, c2 = _fresh()
    prune_model(m2, l2, c2, method, 0.5, log=lambda *a, **k: None,
                prunen=2, prunem=4, group_size=2)
    for n in l1:
        assert torch.equal(l1[n].weight.data, l2[n].weight.data), n


def test_n_m_really_produced_an_n_m_pattern():
    """Equivalence between two wrong results would also pass.

    Exactly two zeros in every run of four along the input dimension.
    """
    m, layers, calib = _fresh()
    prune_model(m, layers, calib, "wanda", 0.5, log=lambda *a, **k: None,
                prunen=2, prunem=4, group_size=2)
    for n, lyr in layers.items():
        w = lyr.weight.data
        blocks = w.reshape(w.shape[0], -1, 4)
        zeros = (blocks == 0).sum(dim=2)
        assert int(zeros.min()) == 2 and int(zeros.max()) == 2, (
            "%s: runs of four hold %d..%d zeros, not 2"
            % (n, int(zeros.min()), int(zeros.max())))
