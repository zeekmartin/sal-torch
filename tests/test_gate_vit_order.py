"""Paper mode has to walk the training data in exp23's order, epoch by epoch.

`_BlockShuffle` creates its generator once and advances it across epochs, so
exp23's second epoch has a different order from its first, and it gets that by
building ONE loader and iterating it twice. A gate that rebuilt the loader per
epoch would repeat the first epoch's order, and `recover_model`'s one-batch
peek at the start would shift every order after it.

Both of those are compared here against a direct exp23-style loop over the
same split with the same seed, which is the only reference that settles it.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "experiments"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gate_v070_vit import NormalizedSplit  # noqa: E402


@pytest.fixture(scope="module")
def proto():
    import run_sal_qat_prototype as P
    return P


def _split(n=64, c=3, hw=4):
    torch.manual_seed(0)
    x = torch.arange(n, dtype=torch.uint8).view(n, 1, 1, 1).expand(n, c, hw, hw)
    y = torch.arange(n)
    return x.contiguous(), y


def _ids(batches):
    """The label of every sample, in the order it was served."""
    out = []
    for b in batches:
        out.extend(b["labels"].tolist() if isinstance(b, dict) else b[1].tolist())
    return out


def _exp23_order(proto, split, bs, seed, block, epochs):
    """One loader, iterated once per epoch: exactly what exp23's recover does."""
    loader = proto.batches(split, bs, shuffle=True, seed=seed, workers=0,
                           block=block)
    return [[y for _x, ys in [(None, b[1]) for b in [bb]] for y in ys.tolist()]
            for _ in range(epochs) for bb in loader]


def _flat_exp23(proto, split, bs, seed, block, epochs):
    loader = proto.batches(split, bs, shuffle=True, seed=seed, workers=0,
                           block=block)
    out = []
    for _ in range(epochs):
        for _x, y in loader:
            out.extend(y.tolist())
    return out


def test_the_second_epoch_differs_from_the_first_as_exp23_does(proto):
    split = _split()
    seen = _flat_exp23(proto, split, 4, seed=42, block=8, epochs=2)
    first, second = seen[:64], seen[64:]
    assert sorted(first) == sorted(second) == list(range(64))
    assert first != second, (
        "the reference itself must reshuffle, or this test proves nothing")


def test_the_gate_reproduces_exp23s_order_including_the_peek(proto):
    """The access pattern recover_model has: one peek, then epoch after epoch."""
    split = _split()
    reference = _flat_exp23(proto, split, 4, seed=42, block=8, epochs=2)

    ns = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                         seed=42, block=8)
    # The peek: one batch from a throwaway pass, exactly as _steps_per_epoch
    # takes it.
    peek = next(iter(ns), None)
    assert peek is not None

    got = []
    for _epoch in range(2):
        got.extend(_ids(ns))
    assert got == reference, (
        "the gate's order diverges from exp23's; paper mode would not be "
        "measuring the published run")


def test_without_the_peek_guard_the_order_would_shift(proto):
    """Show the failure the guard prevents, rather than asserting it is gone."""
    split = _split()
    reference = _flat_exp23(proto, split, 4, seed=42, block=8, epochs=2)

    naive = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                            seed=42, block=8)
    naive._peeked = True          # as if the peek had gone to the real loader
    next(iter(naive), None)
    got = []
    for _epoch in range(2):
        got.extend(_ids(naive))
    assert got != reference, (
        "if this matches, the peek is harmless and the guard is unnecessary")


def test_a_fresh_loader_per_epoch_would_repeat_the_first_epoch(proto):
    """The other failure: rebuilding the loader each epoch."""
    split = _split()
    a = _ids(proto.batches(split, 4, shuffle=True, seed=42, workers=0, block=8))
    b = _ids(proto.batches(split, 4, shuffle=True, seed=42, workers=0, block=8))
    # Two independently built loaders with the same seed give the same order.
    assert [y for y in a] == [y for y in b]


def test_the_evaluation_split_is_not_shuffled(proto):
    split = _split(n=16)
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0)
    first = _ids(ns)
    assert first == list(range(16))
    assert _ids(ns) == first, "an eval pass must be the same every time"
