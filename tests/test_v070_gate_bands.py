"""The gate's bands, exercised. A band nobody can fail is not a gate.

Every test here calls `check()`. For each band the archived values are fed in
and must pass, and a value outside it must fail, so the criterion is known to
discriminate before a GPU-hour is spent on it. The bands themselves are the
pre-registration and are not recomputed here: these tests pin the logic that
reads them, and the numbers they are compared against are typed out from the
published cells on purpose, so a silent edit to the band constants fails.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from v070_gate_bands import (BANDS, DECLARED_PPL_TOLERANCE, RERUN_FLOOR_PP,  # noqa: E402
                             RUN_SETS, check, check_all)

#: The published cells, typed from the write-ups rather than imported from the
#: module under test, so that an edit to the band constants shows up here.
ARCHIVED = {
    "G1-paper": ([78.838, 78.668, 78.942], -2.738),
    "G1-default": ([78.838, 78.668, 78.942], -2.738),
    "G2-paper": ([77.44, 77.642, 77.81], -3.91),
    "G2-default": ([77.44, 77.642, 77.81], -3.91),
    "G3-paper": ([21.8833, 21.7828, 22.3740], 6.7495),
    "G3-default": ([21.8833, 21.7828, 22.3740], 6.7495),
}


def test_there_are_six_bands_over_four_run_sets():
    assert len(BANDS) == 6
    assert sum(len(v) for v in RUN_SETS.values()) == 6
    assert len(RUN_SETS) == 4
    assert {n for names in RUN_SETS.values() for n in names} == set(BANDS)


@pytest.mark.parametrize("name", sorted(ARCHIVED))
def test_the_archived_cell_passes_its_own_band(name):
    per_seed, cost = ARCHIVED[name]
    v = check(name, per_seed, cost)
    assert v["passed"], v["why"]


@pytest.mark.parametrize("name", sorted(ARCHIVED))
def test_a_run_far_outside_fails(name):
    """Shifted by ten times the tolerance, in the direction that is worse."""
    per_seed, cost = ARCHIVED[name]
    band = BANDS[name]
    shift = 10 * (RERUN_FLOOR_PP[1] if band.metric.endswith("%")
                  else DECLARED_PPL_TOLERANCE)
    worse = -shift if band.higher_is_better else shift
    v = check(name, [x + worse for x in per_seed],
              (cost + worse) if band.higher_is_better else (cost + shift))
    assert not v["passed"]
    assert v["why"] and "mean" in " ".join(v["why"])


@pytest.mark.parametrize("name", ["G1-paper", "G2-paper", "G3-paper"])
def test_a_paper_band_also_fails_a_result_that_is_better(name):
    """Paper mode is a reproduction, so it fails on either side."""
    per_seed, cost = ARCHIVED[name]
    band = BANDS[name]
    shift = 10 * (RERUN_FLOOR_PP[1] if band.metric.endswith("%")
                  else DECLARED_PPL_TOLERANCE)
    better = shift if band.higher_is_better else -shift
    v = check(name, [x + better for x in per_seed], cost + better)
    assert not v["passed"], (
        "a two-sided band must fail high as well as low, or it is not a "
        "reproduction check")


@pytest.mark.parametrize("name", ["G1-default", "G2-default", "G3-default"])
def test_a_default_band_passes_a_result_that_is_better(name):
    per_seed, _ = ARCHIVED[name]
    band = BANDS[name]
    better = 1.0 if band.higher_is_better else -1.0
    v = check(name, [x + better for x in per_seed])
    assert v["passed"], v["why"]


@pytest.mark.parametrize("name", ["G1-default", "G2-default", "G3-default"])
def test_a_default_band_fails_a_result_that_is_worse(name):
    per_seed, _ = ARCHIVED[name]
    band = BANDS[name]
    worse = -1.0 if band.higher_is_better else 1.0
    assert not check(name, [x + worse for x in per_seed])["passed"]


def test_one_bad_seed_fails_a_two_sided_band_even_with_a_good_mean():
    """The seeds are checked individually, not only through their mean."""
    good, cost = ARCHIVED["G1-paper"]
    spread = [good[0] - 1.0, good[1], good[2] + 1.0]
    assert abs(sum(spread) / 3 - sum(good) / 3) < 1e-9, "the mean is unchanged"
    v = check("G1-paper", spread, cost)
    assert not v["passed"]
    assert any("seed" in w for w in v["why"])


def test_a_two_sided_band_refuses_a_run_with_no_cost():
    """No matched control, no isolated cost, no pass."""
    per_seed, _ = ARCHIVED["G1-paper"]
    v = check("G1-paper", per_seed, None)
    assert not v["passed"]
    assert any("not isolated" in w for w in v["why"])


def test_a_one_sided_band_does_not_require_a_cost():
    per_seed, _ = ARCHIVED["G1-default"]
    assert check("G1-default", per_seed, None)["passed"]


def test_no_seeds_is_a_failure_not_a_pass():
    v = check("G1-paper", [], -2.738)
    assert not v["passed"] and "no seeds" in v["why"][0]


def test_the_decoder_two_sided_tolerance_is_marked_declared():
    tol = BANDS["G3-paper"].tolerance
    assert "DECLARED" in tol or "declared" in tol
    assert "no decoder cell" in tol or "no measured floor" in tol
    assert BANDS["G3-paper"].seed_hi - max(ARCHIVED["G3-paper"][0]) == \
        pytest.approx(DECLARED_PPL_TOLERANCE)


def test_a_one_sided_band_is_not_widened_at_all():
    """Widening a one-sided bound would only ever weaken it."""
    assert BANDS["G3-default"].mean_hi == max(ARCHIVED["G3-default"][0])
    assert BANDS["G1-default"].mean_lo == min(ARCHIVED["G1-default"][0])
    assert BANDS["G2-default"].mean_lo == min(ARCHIVED["G2-default"][0])
    for name in ("G1-default", "G2-default", "G3-default"):
        assert BANDS[name].seed_lo is None and BANDS[name].seed_hi is None


def test_the_vision_tolerance_names_the_measured_floor():
    assert "measured same-seed rerun floor" in BANDS["G1-paper"].tolerance
    assert RERUN_FLOOR_PP == (0.040, 0.170)


def test_every_band_cites_the_archive():
    for name, band in BANDS.items():
        assert band.sources, name
        for src in band.sources:
            assert src.startswith(("data/", "tables/", "campaign/")), (name, src)


def test_check_all_is_incomplete_until_every_band_is_scored():
    partial = {n: {"per_seed": v[0], "cost": v[1]}
               for n, v in list(ARCHIVED.items())[:2]}
    out = check_all(partial)
    assert out["passed"] is True
    assert out["complete"] is False, (
        "a partial gate must not read as the gate having passed")

    full = {n: {"per_seed": v[0], "cost": v[1]} for n, v in ARCHIVED.items()}
    out = check_all(full)
    assert out["passed"] and out["complete"]
    assert out["n_checked"] == out["n_bands"] == 6


def test_an_unknown_band_raises():
    with pytest.raises(KeyError, match="unknown band"):
        check("G9-paper", [1.0])
