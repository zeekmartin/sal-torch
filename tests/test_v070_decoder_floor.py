"""The decoder band's tolerance was DECLARED; this is the rule for measuring it.

G3's per-seed widening is 0.10 perplexity because no decoder cell in this
campaign was ever re-run at a fixed seed. The vision bands could use a
measured floor (0.040-0.170pp over four dense-control pairs); the decoder had
nothing. So G3-paper failing by 0.2040 on one seed, on the BETTER side, may be
a recipe difference or may be inside what exp36's own script does when you run
it twice, and nothing on disk could tell those apart.

The rule here was pre-registered before the re-run that measures the floor
landed, which is the only time it can be written honestly. Every test runs it.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "experiments"),):
    if p not in sys.path:
        sys.path.insert(0, p)

import score_v070_gate as S  # noqa: E402
from v070_gate_bands import (BANDS, BANDS_WITH_A_DECLARED_TOLERANCE,  # noqa: E402
                             EXPLAINED_BY_NOISE, check, explain_by_noise,
                             measured_floor)

#: What the gate actually measured, in the order the files were read.
G3_SEEDS = [123, 42, 456]
G3_VALUES = [22.0350, 21.4790, 22.4254]
#: Deviation of the one out-of-band seed, 42, from its own archived 21.8833.
S42_DEVIATION = 0.4043


_UNSET = object()


def _verdict(values=None, seeds=_UNSET, cost=6.7159):
    # `seeds=[]` has to reach check() as an empty list, so a falsy default
    # cannot be spelled `seeds or G3_SEEDS`.
    return check("G3-paper", values or G3_VALUES, cost,
                 seeds=(G3_SEEDS if seeds is _UNSET else seeds))


def test_the_gate_result_fails_and_the_failing_seed_is_42():
    v = _verdict()
    assert not v["passed"]
    assert v["seeds"] == G3_SEEDS
    assert any("21.4790" in w for w in v["why"])


def test_a_floor_above_the_deviation_explains_it_and_unblocks_the_tag():
    band = BANDS["G3-paper"]
    out = explain_by_noise(_verdict(), band, floor=0.45,
                           floor_source="job 23898426")
    assert out["status"] == EXPLAINED_BY_NOISE
    assert out["explained"] and not out["blocking"]
    assert not out["passed"], "the band is not widened and does not pass"
    assert "job 23898426" in " ".join(out["why"])
    assert "0.4043" in " ".join(out["why"])


def test_a_floor_below_the_deviation_leaves_it_blocking():
    out = explain_by_noise(_verdict(), BANDS["G3-paper"], floor=0.20)
    assert out["status"] == "FAIL" and out["blocking"]
    assert "EXCEEDS" in " ".join(out["why"])


def test_without_a_measured_floor_the_declared_tolerance_stands():
    out = explain_by_noise(_verdict(), BANDS["G3-paper"], floor=None)
    assert out["blocking"] and not out["explained"]
    assert "no measured same-seed floor" in " ".join(out["why"])


def test_a_mean_out_of_band_is_never_explained_by_single_seed_noise():
    """One seed can be noise; a mean that moves is not."""
    v = _verdict(values=[25.0, 25.0, 25.0])
    out = explain_by_noise(v, BANDS["G3-paper"], floor=99.0)
    assert out["blocking"]
    assert "not explainable by single-seed noise" in " ".join(out["why"])


def test_a_cost_out_of_band_is_not_explained_either():
    v = _verdict(cost=99.0)
    out = explain_by_noise(v, BANDS["G3-paper"], floor=99.0)
    assert out["blocking"]


def test_the_rule_does_not_touch_a_vision_band():
    """Their widening already contains a measured floor; twice would be wrong."""
    assert BANDS_WITH_A_DECLARED_TOLERANCE == ("G3-paper",)
    v = check("G1-paper", [78.404, 78.606, 78.792], -2.9533,
              seeds=[123, 42, 456])
    out = explain_by_noise(v, BANDS["G1-paper"], floor=99.0)
    assert out["blocking"] and not out["explained"]


def test_a_passing_band_is_left_alone():
    v = check("G3-default", [21.6836, 21.5016, 22.4156], 6.6031,
              seeds=G3_SEEDS)
    assert v["passed"]
    out = explain_by_noise(v, BANDS["G3-default"], floor=0.45)
    assert out["status"] == "PASS" and not out["blocking"]


def test_the_deviation_is_matched_by_seed_and_not_by_position():
    """The cells arrive as 123, 42, 456. Zipping would compare the wrong pair."""
    band = BANDS["G3-paper"]
    by_seed = band.archived_by_seed
    positional = measured_floor(G3_VALUES[0], band.archived_seeds[0])
    correct = measured_floor(G3_VALUES[0], by_seed[G3_SEEDS[0]])
    assert abs(positional - correct) > 0.1, (
        "if these agree the fixture no longer demonstrates the hazard")
    out = explain_by_noise(_verdict(), band, floor=0.45)
    assert "seed 42 0.4043" in " ".join(out["why"])


def test_seeds_the_archive_does_not_have_are_refused():
    out = explain_by_noise(_verdict(seeds=[1, 2, 3]), BANDS["G3-paper"],
                           floor=0.45)
    assert out["blocking"]
    assert "no value for seed" in " ".join(out["why"])


def test_unlabelled_seeds_are_refused_rather_than_zipped():
    out = explain_by_noise(_verdict(seeds=[]), BANDS["G3-paper"], floor=0.45)
    assert out["blocking"]
    assert "cannot be matched to the archive by seed" in " ".join(out["why"])


# --------------------------------------------------------- reading the floor
def _arm_file(tmp_path, name, one_shot, recovered, seed=42):
    p = tmp_path / name
    p.write_text(json.dumps({"arms": {"heads_random_50": {"seeds": {
        str(seed): {"oneshot_wikitext2_ppl": one_shot,
                    "recovered_wikitext2_ppl": recovered}}}}}),
        encoding="utf-8")
    return str(p)


def test_the_floor_is_read_from_both_runs_not_typed(tmp_path):
    arch = _arm_file(tmp_path, "archive.json", 93.5936, 21.8833)
    rerun = _arm_file(tmp_path, "rerun.json", 93.5936, 22.0501)
    floor, source = S.decoder_floor(rerun, arch)
    assert abs(floor - 0.1668) < 1e-9
    assert "21.8833" in source and "22.0501" in source


def test_a_rerun_that_removed_other_heads_is_void(tmp_path):
    arch = _arm_file(tmp_path, "archive.json", 93.5936, 21.8833)
    rerun = _arm_file(tmp_path, "rerun.json", 75.2705, 22.0501)
    with pytest.raises(ValueError, match="did not remove the same heads"):
        S.decoder_floor(rerun, arch)


def test_a_rerun_missing_the_seed_raises(tmp_path):
    arch = _arm_file(tmp_path, "archive.json", 93.5936, 21.8833)
    rerun = _arm_file(tmp_path, "rerun.json", 93.5936, 22.05, seed=123)
    with pytest.raises(KeyError):
        S.decoder_floor(rerun, arch)
