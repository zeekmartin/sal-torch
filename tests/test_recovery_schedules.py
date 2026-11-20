"""Paper mode must reproduce the schedule the archived numbers ran under.

``recover(schedule="one_cycle")`` carries its own copy of ``OneCycleLR``'s
arithmetic, and ``schedule="warmup_cosine"`` carries the vision recipe's. A
copy is only worth having if it agrees with the original, so both are stepped
side by side here against the thing they copy, at the campaign's real step
counts, and required to agree to 1e-12 relative at every step.

The references are the real ones, not reproductions of them: torch's
``OneCycleLR`` for the decoder shape, and ``scripts/run_sal_qat_prototype.py``
``make_optimizer`` for the vision shape, which is the function that produced
every published vision cell.
"""
import math

import pytest
import torch

from sal.recovery import (ONE_CYCLE_DIV_FACTOR, ONE_CYCLE_FINAL_DIV_FACTOR,
                          _one_cycle, _warmup_cosine)

#: OPT-1.3B recovery as it ran: 32M tokens at batch 4 x 2048 tokens is
#: 32_000_000 / 8192 = 3906.25, so 3906 steps, max_lr 1e-5, pct_start 0.10.
#: Public data archive, DOI 10.5281/zenodo.24391200,
#: data/pro/results/EXP36_OPT_SPARSE.md.
OPT_STEPS, OPT_MAX_LR, OPT_PCT_START = 3906, 1e-5, 0.10

#: ViT-B/16 recovery as it ran: 1_281_167 images at batch 64 is 20019 steps an
#: epoch, two epochs, so 40038 steps.
VIT_STEPS = 40038


def _torch_one_cycle_lrs(total_steps, max_lr, pct_start):
    """The learning rate torch hands the optimizer at every step of the run."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=max_lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start,
        anneal_strategy="cos", div_factor=ONE_CYCLE_DIV_FACTOR,
        final_div_factor=ONE_CYCLE_FINAL_DIV_FACTOR)
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    return lrs


def _ours(total_steps, max_lr, pct_start):
    factor, peak = _one_cycle(total_steps, pct_start)
    return [max_lr * factor(i) for i in range(total_steps)], peak


# ------------------------------------------------------- the decoder schedule
def test_one_cycle_matches_torch_at_the_opt_recovery_size():
    theirs = _torch_one_cycle_lrs(OPT_STEPS, OPT_MAX_LR, OPT_PCT_START)
    ours, peak = _ours(OPT_STEPS, OPT_MAX_LR, OPT_PCT_START)
    assert len(ours) == len(theirs) == OPT_STEPS
    worst = max(abs(a - b) / b for a, b in zip(ours, theirs))
    assert worst < 1e-12, f"worst relative difference {worst:.3e}"
    # And the shape that matters: it is still climbing at step 300 of 3906.
    assert peak == 389
    assert ours[0] == pytest.approx(OPT_MAX_LR / ONE_CYCLE_DIV_FACTOR, rel=1e-12)
    assert ours[300] > ours[0] and ours[300] < max(ours)


@pytest.mark.parametrize("total_steps,pct_start", [
    (3906, 0.10),      # the OPT recovery
    (40038, 0.10),     # the same schedule at the vision step count
    (1000, 0.25),
    (100, 0.03),       # up phase ends at 2.0
    (37, 0.5),
    (11, 0.1),         # up phase ends at 0.1, i.e. before step 0
])
def test_one_cycle_matches_torch_across_sizes(total_steps, pct_start):
    theirs = _torch_one_cycle_lrs(total_steps, 1e-5, pct_start)
    ours, _ = _ours(total_steps, 1e-5, pct_start)
    worst = max(abs(a - b) / b for a, b in zip(ours, theirs))
    assert worst < 1e-12, f"worst relative difference {worst:.3e}"


def test_the_edge_case_torch_cannot_do():
    """``pct_start * total_steps == 1`` divides by zero inside OneCycleLR.

    That is the whole reason the arithmetic is carried here: it is what a unit
    test on a tiny fixture produces, and a schedule that cannot run on the
    fixture cannot be tested. Ours returns the peak at step 0 and goes on.
    """
    with pytest.raises(ZeroDivisionError):
        _torch_one_cycle_lrs(10, 1e-5, 0.10)

    factor, peak = _one_cycle(10, 0.10)
    assert peak == 0
    assert factor(0) == pytest.approx(1.0, rel=1e-12)      # the peak, at once
    assert factor(1) < factor(0)                            # and decaying after
    lrs = [factor(i) for i in range(10)]
    assert lrs == sorted(lrs, reverse=True)

    # One step past the end is NOT clamped, exactly as torch does not clamp it:
    # the cosine turns back upward there. torch forbids that step with a
    # ValueError; recover_model simply does not take it, which is why the
    # recorded curve ends at the floor rather than above it.
    assert factor(10) > factor(9)


def test_one_cycle_ends_near_the_final_floor():
    factor, _ = _one_cycle(OPT_STEPS, OPT_PCT_START)
    floor = 1.0 / ONE_CYCLE_DIV_FACTOR / ONE_CYCLE_FINAL_DIV_FACTOR
    assert factor(OPT_STEPS - 1) == pytest.approx(floor, rel=1e-6)


# -------------------------------------------------------- the vision schedule
def test_warmup_cosine_matches_the_published_vision_recipe():
    """Against ``make_optimizer`` itself, which produced the published cells."""
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import run_sal_qat_prototype as P
    finally:
        sys.path.remove(str(scripts))

    assert P.WARMUP_FRACTION == 0.05, "the recipe this pins has changed"
    model = torch.nn.Linear(4, 4)
    opt, sched = P.make_optimizer(model, VIT_STEPS)
    theirs = []
    for _ in range(VIT_STEPS):
        theirs.append(opt.param_groups[0]["lr"])
        sched.step()

    factor, warmup = _warmup_cosine(VIT_STEPS, P.WARMUP_FRACTION)
    ours = [P.LR_BACKBONE * factor(i) for i in range(VIT_STEPS)]

    worst = max(abs(a - b) / b for a, b in zip(ours, theirs))
    assert worst < 1e-12, f"worst relative difference {worst:.3e}"
    assert warmup == max(1, int(P.WARMUP_FRACTION * VIT_STEPS)) == 2001
    # The property the product default is chosen for: the peak is inside the
    # warmup window and the rate never exceeds the rate that was asked for.
    assert ours.index(max(ours)) <= warmup
    assert max(ours) == pytest.approx(P.LR_BACKBONE, rel=1e-12)


def test_warmup_cosine_shape_at_the_edges():
    factor, warmup = _warmup_cosine(100, 0.05)
    assert warmup == 5
    assert factor(0) == pytest.approx(1 / 5)
    assert factor(4) == pytest.approx(1.0)
    assert factor(99) < 1e-3
    assert factor(99) == pytest.approx(
        0.5 * (1 + math.cos(math.pi * (99 - 5) / 95)), rel=1e-12)

    # A one-step run still has a warmup of one step rather than zero.
    f1, w1 = _warmup_cosine(1, 0.05)
    assert w1 == 1 and f1(0) == pytest.approx(1.0)
