"""The matched-control helper: it must reproduce the published costs and
refuse every pairing the convention forbids.

CPU-only, no network. Everything is JSON plus arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from matched_control import (  # noqa: E402
    PRETRAINED, PUBLISHED, Control, ControlError, ControlLadder, cost_table)

# What EXP32_RATIO_AT_2EP.md publishes, three seeds, matched 2-epoch control.
PUBLISHED_COSTS = {0.80258: -1.30, 0.79626: -1.93, 0.77642: -3.91}
PUBLISHED_COSTS_INT4 = {0.78744: -1.53, 0.78003: -2.27, 0.75427: -4.85}


@pytest.fixture(scope="module")
def ladder():
    return ControlLadder.load()


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #

def test_ladder_loads_and_verifies(ladder):
    assert set(ladder.controls) == {2, 5}
    report = ladder.verify()
    assert report[2]["checked"] and report[2]["seeds"] == 3
    assert report[5]["checked"] and report[5]["seeds"] == 1


def test_two_epoch_control_is_the_published_mean(ladder):
    c = ladder.at(2)
    assert c.accuracy == pytest.approx(0.81554, abs=5e-6)
    assert c.int4_accuracy == pytest.approx(0.80273, abs=5e-6)
    assert c.seeds == 3


def test_per_seed_controls_are_the_published_three(ladder):
    got = sorted(round(ladder.at(2, seed=s).accuracy * 100, 3)
                 for s in (42, 123, 456))
    assert got == [81.462, 81.524, 81.676]


def test_fine_tuning_gain_is_control_minus_pretrained(ladder):
    c = ladder.at(2)
    assert c.fine_tuning_gain_pp == pytest.approx(
        (c.accuracy - PRETRAINED["accuracy"]) * 100, abs=1e-9)
    assert c.fine_tuning_gain_pp == pytest.approx(3.114, abs=0.01)
    assert ladder.at(5).fine_tuning_gain_pp == pytest.approx(2.116, abs=0.01)


def test_asking_for_a_missing_seed_names_the_ones_it_has(ladder):
    with pytest.raises(ControlError, match=r"seed 999.*\[42, 123, 456\]"):
        ladder.at(2, seed=999)


def test_asking_for_a_missing_epoch_count_lists_what_exists(ladder):
    with pytest.raises(ControlError, match=r"no 3-epoch control"):
        ladder.at(3)


def test_a_control_file_whose_epochs_disagree_is_refused(tmp_path, monkeypatch):
    """The registry says 2 epochs; if the data says otherwise, stop."""
    import json

    import matched_control as mc

    bad = tmp_path / "ctrl.json"
    bad.write_text(json.dumps({
        "epochs": 7, "seed": 42,
        "pretrained": PRETRAINED,
        "control": {"accuracy": 0.81554, "int4_accuracy": 0.80273, "epochs": 7},
    }))
    monkeypatch.setitem(mc.CONTROL_SOURCES, 2, [str(bad)])
    with pytest.raises(ControlError, match="says epochs=7"):
        ControlLadder.load(epochs=(2,), root=Path("/"))


def test_a_drifted_control_raises_rather_than_redefining_every_cost(tmp_path,
                                                                    monkeypatch):
    import json

    import matched_control as mc

    drifted = tmp_path / "ctrl.json"
    drifted.write_text(json.dumps({
        "epochs": 2, "seed": 42,
        "pretrained": PRETRAINED,
        # 1.5pp below the published 81.554, far outside the 0.25pp tolerance
        "control": {"accuracy": 0.80054, "int4_accuracy": 0.80273, "epochs": 2},
    }))
    monkeypatch.setitem(mc.CONTROL_SOURCES, 2, [str(drifted)])
    with pytest.raises(ControlError, match="against the published"):
        ControlLadder.load(epochs=(2,), root=Path("/"))


def test_verify_can_be_skipped_but_is_on_by_default(tmp_path, monkeypatch):
    import json

    import matched_control as mc

    drifted = tmp_path / "ctrl.json"
    drifted.write_text(json.dumps({
        "epochs": 2, "seed": 42, "pretrained": PRETRAINED,
        "control": {"accuracy": 0.70000, "int4_accuracy": 0.69, "epochs": 2}}))
    monkeypatch.setitem(mc.CONTROL_SOURCES, 2, [str(drifted)])
    ladder = ControlLadder.load(epochs=(2,), verify=False, root=Path("/"))
    assert ladder.at(2).accuracy == pytest.approx(0.70)
    with pytest.raises(ControlError):
        ladder.verify()


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm_acc,expected", PUBLISHED_COSTS.items())
def test_matched_costs_reproduce_the_published_table(ladder, arm_acc, expected):
    row = ladder.at(2).cost(arm_acc, arm_epochs=2)
    assert row["epochs_matched"] is True
    assert row["pruning_cost_pp"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("arm_acc,expected", PUBLISHED_COSTS_INT4.items())
def test_matched_int4_costs_reproduce_the_published_table(ladder, arm_acc,
                                                          expected):
    row = ladder.at(2).cost(0.8, arm_epochs=2, arm_int4_accuracy=arm_acc)
    assert row["pruning_cost_int4_pp"] == pytest.approx(expected, abs=0.01)


def test_an_unmatched_pairing_emits_no_pruning_cost(ladder):
    row = ladder.at(2).cost(0.77265, arm_epochs=5)
    assert row["epochs_matched"] is False
    assert "pruning_cost_pp" not in row
    assert "pruning_cost_int4_pp" not in row
    assert "5 epochs and this control ran 2" in row["why_no_pruning_cost"]


def test_the_exp31_bug_cannot_be_reproduced_through_this_helper(ladder):
    """-2.269 / -2.831 / -4.411 were 5-epoch arms on a 2-epoch control."""
    for arm in (0.79407, 0.78845, 0.77265):
        row = ladder.at(2, seed=42).cost(arm, arm_epochs=5)
        assert "pruning_cost_pp" not in row


def test_the_five_epoch_arms_do_get_a_cost_from_the_five_epoch_control(ladder):
    row = ladder.at(5).cost(0.77265, arm_epochs=5)
    assert row["epochs_matched"] is True
    assert row["pruning_cost_pp"] == pytest.approx(-3.291, abs=0.01)


def test_vs_pretrained_is_always_labelled_as_a_sum(ladder):
    for epochs in (2, 5):
        row = ladder.at(2).cost(0.8, arm_epochs=epochs)
        assert "SUM" in row["vs_pretrained_is"]
        assert "not a pruning cost" in row["vs_pretrained_is"]


def test_fine_tuning_gain_is_reported_even_when_unmatched(ladder):
    """The two effects are separate; one being unavailable does not hide the other."""
    row = ladder.at(2).cost(0.77265, arm_epochs=5)
    assert row["fine_tuning_gain_pp"] == pytest.approx(3.114, abs=0.01)


def test_within_seed_scoring_differs_from_the_mean_and_is_available(ladder):
    arm = 0.77810                                   # exp32 seed 42's P50 cell
    within = ladder.at(2, seed=42).cost(arm, arm_epochs=2)["pruning_cost_pp"]
    against_mean = ladder.at(2).cost(arm, arm_epochs=2)["pruning_cost_pp"]
    assert within != pytest.approx(against_mean, abs=1e-6)
    assert within == pytest.approx((arm - 0.81676) * 100, abs=1e-9)


def test_int4_cost_is_absent_when_the_control_has_no_int4():
    c = Control(epochs=2, accuracy=0.8, int4_accuracy=None, seed=1)
    row = c.cost(0.78, arm_epochs=2, arm_int4_accuracy=0.77)
    assert "pruning_cost_int4_pp" not in row
    assert row["fine_tuning_gain_int4_pp"] is None


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def test_cost_table_sets_unmatched_rows_apart(ladder):
    c = ladder.at(2)
    text = cost_table([
        c.cost(0.77642, arm_epochs=2, arm_int4_accuracy=0.75427, label="matched"),
        c.cost(0.77265, arm_epochs=5, label="unmatched"),
    ], title="t")
    assert "-3.91" in text
    assert "n/a" in text
    assert "a 2-epoch control is needed" in text


def test_to_dict_carries_the_convention_and_the_verification(ladder):
    d = ladder.to_dict()
    assert d["convention"].endswith("CONTROL_CONVENTION.md")
    assert d["verified"][2]["checked"] is True
    assert d["controls"]["2"]["seeds"] == 3
    assert d["pretrained"]["accuracy"] == 0.7844


def test_published_registry_matches_the_loaded_corpus(ladder):
    """If someone re-runs a control, this is the test that notices."""
    for ep, pub in PUBLISHED.items():
        assert ladder.at(ep).seeds == pub["seeds"]
        assert ladder.at(ep).accuracy == pytest.approx(
            pub["mean"], abs=pub["tolerance_pp"] / 100)
