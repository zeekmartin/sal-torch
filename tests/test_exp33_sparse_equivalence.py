"""Layer-level equivalence of the vendored SparseGPT / Wanda against upstream.

Two tiers, because the reference implementations are third-party repos we do
not vendor:

1. **Invariants that always run.** Exact sparsity, which weights may move,
   idempotence, and the metric ordering the mask has to agree with. These are
   the properties the experiment's conclusions rest on, and they run in CI.
2. **The equivalence test proper**, which compares against the upstream code
   and is *skipped unless* ``SAL_SPARSE_REFS`` points at a directory holding
   clones of ``IST-DASLab/sparsegpt`` and ``locuslab/wanda``::

       git clone --depth 1 https://github.com/IST-DASLab/sparsegpt  $REFS/sparsegpt
       git clone --depth 1 https://github.com/locuslab/wanda        $REFS/wanda
       SAL_SPARSE_REFS=$REFS python -m pytest tests/test_exp33_sparse_equivalence.py

Why not compare against the papers' numbers: those are LLM perplexities on
WikiText, measured on models and a calibration pipeline we are not running.
A matching perplexity would not show our ViT pruning pass is right, and a
differing one would not show it is wrong. The layer-level comparison is the
check that actually bears on correctness.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp33_sparse_core import SparseGPTPruner, WandaPruner  # noqa: E402

REFS = os.environ.get("SAL_SPARSE_REFS")
needs_refs = pytest.mark.skipif(
    not REFS or not Path(REFS).is_dir(),
    reason="set SAL_SPARSE_REFS to a dir holding sparsegpt/ and wanda/ clones")

ROWS, COLS, NCALIB = 48, 64, 16
SPARSITY = 0.5


@pytest.fixture
def layer_and_calib():
    """A linear layer and a fixed calibration batch, identical every run."""
    torch.manual_seed(0)
    layer = nn.Linear(COLS, ROWS, bias=False)
    calib = torch.randn(NCALIB, 7, COLS)      # (batch, tokens, in_features)
    return layer, calib


def _fresh(layer):
    import copy
    return copy.deepcopy(layer)


# ------------------------------------------------------------- tier 1: ours
def test_wanda_zeroes_exactly_the_requested_fraction_per_row(layer_and_calib):
    layer, calib = layer_and_calib
    pr = WandaPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    pr.prune(SPARSITY)
    per_row = (layer.weight.data == 0).sum(dim=1)
    assert per_row.tolist() == [int(COLS * SPARSITY)] * ROWS


def test_wanda_mask_agrees_with_the_metric_ordering(layer_and_calib):
    """The kept weights must all out-score the dropped ones, within each row."""
    layer, calib = layer_and_calib
    before = layer.weight.data.clone()
    pr = WandaPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    metric = before.abs() * torch.sqrt(pr.scaler_row.reshape(1, -1))
    pr.prune(SPARSITY)
    dropped = layer.weight.data == 0
    for r in range(ROWS):
        assert metric[r][dropped[r]].max() <= metric[r][~dropped[r]].min() + 1e-6


def test_wanda_leaves_surviving_weights_untouched(layer_and_calib):
    """Wanda only masks. Anything it did not zero must be bit-identical."""
    layer, calib = layer_and_calib
    before = layer.weight.data.clone()
    pr = WandaPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    pr.prune(SPARSITY)
    kept = layer.weight.data != 0
    assert torch.equal(layer.weight.data[kept], before[kept])


def test_sparsegpt_hits_the_requested_sparsity_and_updates_survivors(
        layer_and_calib):
    """SparseGPT compensates, so survivors *must* move — that is the method."""
    layer, calib = layer_and_calib
    before = layer.weight.data.clone()
    pr = SparseGPTPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    pr.prune(SPARSITY)
    frac = (layer.weight.data == 0).float().mean().item()
    assert abs(frac - SPARSITY) < 0.02
    kept = layer.weight.data != 0
    assert not torch.equal(layer.weight.data[kept], before[kept])
    assert torch.isfinite(layer.weight.data).all()


def test_sparsegpt_is_deterministic_given_the_calibration_set(layer_and_calib):
    layer, calib = layer_and_calib
    outs = []
    for _ in range(2):
        lyr = _fresh(layer)
        pr = SparseGPTPruner(lyr)
        for i in range(NCALIB):
            pr.add_batch(calib[i])
        pr.prune(SPARSITY)
        outs.append(lyr.weight.data.clone())
    assert torch.equal(outs[0], outs[1])


def test_zero_sparsity_is_a_no_op_for_wanda(layer_and_calib):
    layer, calib = layer_and_calib
    before = layer.weight.data.clone()
    pr = WandaPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    pr.prune(0.0)
    assert torch.equal(layer.weight.data, before)


# ------------------------------------------------- tier 1: 2:4 invariants
def test_wanda_24_zeroes_two_of_every_four(layer_and_calib):
    layer, calib = layer_and_calib
    pr = WandaPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    pr.prune(0.0, prunen=2, prunem=4)
    z = layer.weight.data == 0
    groups = z.reshape(ROWS, COLS // 4, 4).sum(dim=-1)
    assert groups.unique().tolist() == [2], groups.unique().tolist()


def test_sparsegpt_24_zeroes_two_of_every_four(layer_and_calib):
    layer, calib = layer_and_calib
    pr = SparseGPTPruner(layer)
    for i in range(NCALIB):
        pr.add_batch(calib[i])
    pr.prune(0.0, prunen=2, prunem=4)
    z = layer.weight.data == 0
    groups = z.reshape(ROWS, COLS // 4, 4).sum(dim=-1)
    assert groups.unique().tolist() == [2], groups.unique().tolist()
    assert torch.isfinite(layer.weight.data).all()


def test_24_ignores_the_sparsity_argument(layer_and_calib):
    """n:m fixes the fraction, so sparsity must not change the result."""
    layer, calib = layer_and_calib
    outs = []
    for sp in (0.0, 0.9):
        lyr = _fresh(layer)
        pr = WandaPruner(lyr)
        for i in range(NCALIB):
            pr.add_batch(calib[i])
        pr.prune(sp, prunen=2, prunem=4)
        outs.append(lyr.weight.data.clone())
    assert torch.equal(outs[0], outs[1])


# -------------------------------------------- tier 2: against the references
@needs_refs
def test_wanda_matches_the_reference(layer_and_calib):
    """Same layer, same calibration: identical mask, identical weights."""
    sys.path.insert(0, str(Path(REFS) / "wanda" / "image_classifiers"))
    from layerwrapper import WrappedLayer          # noqa: E402
    from prune_utils import compute_mask           # noqa: E402

    layer, calib = layer_and_calib
    ours, theirs = _fresh(layer), _fresh(layer)

    mine = WandaPruner(ours)
    ref = WrappedLayer(theirs)
    for i in range(NCALIB):
        mine.add_batch(calib[i])
        ref.add_batch(calib[i], None)

    assert torch.allclose(mine.scaler_row, ref.scaler_row, atol=1e-6), \
        "activation-norm accumulation differs"

    mine.prune(SPARSITY)
    metric = torch.abs(theirs.weight.data) * torch.sqrt(
        ref.scaler_row.reshape((1, -1)))
    theirs.weight.data[compute_mask(metric, "row", SPARSITY)] = 0

    assert torch.equal(ours.weight.data == 0, theirs.weight.data == 0), \
        "masks differ"
    assert torch.allclose(ours.weight.data, theirs.weight.data, atol=1e-6)


@needs_refs
def test_sparsegpt_matches_the_reference(layer_and_calib, monkeypatch):
    """Same layer, same calibration: identical mask, weights within tolerance.

    ``fasterprune`` ends with an unconditional ``torch.cuda.synchronize()``,
    which raises on a CPU-only torch build ("Torch not compiled with CUDA
    enabled"). It is a barrier, not arithmetic, so the test stubs it out; the
    weights it produces are unaffected. On ROCm the call is live and works,
    since torch maps ``torch.cuda`` onto HIP.
    """
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    sys.path.insert(0, str(Path(REFS) / "sparsegpt"))
    from sparsegpt import SparseGPT                # noqa: E402

    layer, calib = layer_and_calib
    ours, theirs = _fresh(layer), _fresh(layer)

    mine = SparseGPTPruner(ours)
    ref = SparseGPT(theirs)
    for i in range(NCALIB):
        mine.add_batch(calib[i])
        ref.add_batch(calib[i], None)

    assert torch.allclose(mine.H, ref.H, atol=1e-5), "Hessian differs"

    mine.prune(SPARSITY)
    ref.fasterprune(SPARSITY, blocksize=128, percdamp=0.01)

    assert torch.equal(ours.weight.data == 0, theirs.weight.data == 0), \
        "masks differ"
    assert torch.allclose(ours.weight.data, theirs.weight.data, atol=1e-5), \
        (ours.weight.data - theirs.weight.data).abs().max().item()

@needs_refs
def test_wanda_24_matches_the_reference(layer_and_calib):
    """2:4 masks must match upstream's n:m branch exactly."""
    sys.path.insert(0, str(Path(REFS) / "wanda" / "image_classifiers"))
    from layerwrapper import WrappedLayer          # noqa: E402

    layer, calib = layer_and_calib
    ours, theirs = _fresh(layer), _fresh(layer)

    mine = WandaPruner(ours)
    ref = WrappedLayer(theirs)
    for i in range(NCALIB):
        mine.add_batch(calib[i])
        ref.add_batch(calib[i], None)
    mine.prune(0.0, prunen=2, prunem=4)

    # lib/prune.py::prune_wanda, the prune_n != 0 branch, verbatim
    metric = torch.abs(theirs.weight.data) * torch.sqrt(
        ref.scaler_row.reshape((1, -1)))
    W_mask = torch.zeros_like(metric) == 1
    for ii in range(metric.shape[1]):
        if ii % 4 == 0:
            tmp = metric[:, ii:(ii + 4)].float()
            W_mask.scatter_(1, ii + torch.topk(tmp, 2, dim=1,
                                               largest=False)[1], True)
    theirs.weight.data[W_mask] = 0

    assert torch.equal(ours.weight.data == 0, theirs.weight.data == 0)
    assert torch.allclose(ours.weight.data, theirs.weight.data, atol=1e-6)


@needs_refs
def test_sparsegpt_24_matches_the_reference(layer_and_calib, monkeypatch):
    """2:4 through the reference's own prunen/prunem path."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    sys.path.insert(0, str(Path(REFS) / "sparsegpt"))
    from sparsegpt import SparseGPT                # noqa: E402

    layer, calib = layer_and_calib
    ours, theirs = _fresh(layer), _fresh(layer)

    mine = SparseGPTPruner(ours)
    ref = SparseGPT(theirs)
    for i in range(NCALIB):
        mine.add_batch(calib[i])
        ref.add_batch(calib[i], None)

    mine.prune(0.0, prunen=2, prunem=4)
    ref.fasterprune(0.0, prunen=2, prunem=4, blocksize=128, percdamp=0.01)

    assert torch.equal(ours.weight.data == 0, theirs.weight.data == 0), \
        "2:4 masks differ"
    assert torch.allclose(ours.weight.data, theirs.weight.data, atol=1e-5), \
        (ours.weight.data - theirs.weight.data).abs().max().item()
