"""Paper mode has to augment the way the archive augments.

`run_sal_qat_prototype.make_step` moves the batch to the device, normalizes
it, then flips half of it horizontally, one `torch.rand(n, device=...) < 0.5`
draw per training batch. `recover()` does none of that and should not: it has
to work on tokens too. So the gate's vision data view carries it, and these
tests check it against a direct make_step-style loop over the same split with
the same seed, which is the only reference that settles it.

The cost of not having it, measured: training loss 0.049 lower in epoch one
and 0.039 lower in epoch two on 6 of 6 vision runs, spread 0.0006 across three
seeds, and 0.15 to 0.26pp of validation accuracy. That is what failed G1.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "experiments"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gate_v070_vit import AUGMENTATION, NormalizedSplit  # noqa: E402


@pytest.fixture(scope="module")
def proto():
    import run_sal_qat_prototype as P
    return P


def _split(n=32, c=3, hw=4):
    """Width-varying images, or a horizontal flip is undetectable."""
    col = torch.arange(hw, dtype=torch.uint8).view(1, 1, 1, hw)
    idx = torch.arange(n, dtype=torch.uint8).view(n, 1, 1, 1)
    x = (col * 7 + idx).expand(n, c, hw, hw).contiguous()
    return x, torch.arange(n)


def _reference(proto, split, bs, seed, block, epochs, root_seed=0):
    """exp40's own input pipeline: no peek, one flip draw per batch."""
    torch.manual_seed(root_seed)
    loader = proto.batches(split, bs, shuffle=True, seed=seed, workers=0,
                           block=block)
    out = []
    for _ in range(epochs):
        for x, _y in loader:
            x = proto.normalize(x)
            picked = torch.rand(x.shape[0]) < 0.5
            out.append(torch.where(picked.view(-1, 1, 1, 1), x.flip(-1), x))
    return out


def _from_view(ns, epochs, root_seed=0, peek=True):
    torch.manual_seed(root_seed)
    if peek:
        next(iter(ns), None)          # the schedule's peek, as recover() takes it
    out = []
    for _ in range(epochs):
        for b in ns:
            out.append(b["pixel_values"])
    return out


def test_the_view_reproduces_make_steps_augmented_batches(proto):
    split = _split()
    ref = _reference(proto, split, 4, seed=42, block=8, epochs=2)
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                         seed=42, block=8, flip=True)
    got = _from_view(ns, epochs=2)
    assert len(got) == len(ref)
    for i, (a, b) in enumerate(zip(got, ref)):
        assert torch.equal(a, b), "batch %d differs from the archived pipeline" % i


def test_the_augmentation_actually_flips_something(proto):
    """A reference both sides compute identically could still be a no-op."""
    split = _split()
    plain = NormalizedSplit(split, 4, proto.normalize, workers=0)
    flipped = NormalizedSplit(split, 4, proto.normalize, workers=0, flip=True)
    torch.manual_seed(0)
    a = [b["pixel_values"] for b in plain]
    torch.manual_seed(0)
    b = [x["pixel_values"] for x in flipped]
    assert any(not torch.equal(p, q) for p, q in zip(a, b))
    # and every flipped batch is either the image or its mirror, nothing else
    for p, q in zip(a, b):
        for j in range(p.shape[0]):
            assert (torch.equal(p[j], q[j])
                    or torch.equal(p[j].flip(-1), q[j]))


def test_a_draw_on_the_peek_would_shift_every_batch_after_it(proto):
    """Show the failure the guard prevents, rather than asserting it is gone."""
    split = _split()
    ref = _reference(proto, split, 4, seed=42, block=8, epochs=2)
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                         seed=42, block=8, flip=True)
    ns._peeked = True                 # as if the peek had taken its flip draw
    got = _from_view(ns, epochs=2, peek=True)
    assert any(not torch.equal(a, b) for a, b in zip(got, ref)), (
        "if this matches, the peek costs nothing and the guard is unnecessary")


def test_the_evaluation_view_is_never_augmented(proto):
    split = _split(n=16)
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0)
    torch.manual_seed(0)
    first = [b["pixel_values"].clone() for b in ns]
    torch.manual_seed(1)
    again = [b["pixel_values"] for b in ns]
    for a, b in zip(first, again):
        assert torch.equal(a, b), "an eval pass moved with the RNG"


def test_the_order_still_matches_exp23_with_the_flip_on(proto):
    """The augmentation must not disturb the sampler the order tests pin."""
    split = _split()
    labels = []
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                         seed=42, block=8, flip=True)
    torch.manual_seed(0)
    next(iter(ns), None)
    for _ in range(2):
        for b in ns:
            labels.extend(b["labels"].tolist())
    loader = proto.batches(split, 4, shuffle=True, seed=42, workers=0, block=8)
    ref = []
    for _ in range(2):
        for _x, y in loader:
            ref.extend(y.tolist())
    assert labels == ref


def test_the_cell_records_which_augmentation_ran():
    assert "horizontal flip" in AUGMENTATION and "0.5" in AUGMENTATION


def test_the_peek_leaves_the_global_generator_untouched(proto):
    """The second-order bug that the equivalence test caught.

    `iter(DataLoader)` draws a base seed from the GLOBAL generator when the
    loader has none of its own. Serving the peek from a throwaway loader
    therefore left the stream one draw ahead, and every flip decision after it
    differed from exp40's while the data order still matched perfectly. Order
    tests could not see it; this one states the invariant directly.
    """
    split = _split()
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                         seed=42, block=8, flip=True)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    peek = next(iter(ns), None)
    assert peek is not None and peek["pixel_values"].shape[0] == 4
    assert torch.equal(torch.get_rng_state(), before), (
        "the peek consumed randomness, so every batch after it is a different "
        "run from the published one")


def test_the_peek_is_not_augmented_and_is_never_trained_on(proto):
    """Its content is irrelevant, so it must not spend a flip draw either."""
    split = _split()
    ns = NormalizedSplit(split, 4, proto.normalize, workers=0, shuffle=True,
                         seed=42, block=8, flip=True)
    peek = next(iter(ns))
    plain = proto.normalize(split[0][:4])
    assert torch.equal(peek["pixel_values"], plain)
