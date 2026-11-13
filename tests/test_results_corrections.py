"""The corrections sidecar: it must fix derived fields and never touch disk.

CPU-only, no network, no model. Everything here is JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from results_corrections import (  # noqa: E402
    CORRECTIONS_PATH, CorrectionError, audit, load_corrections, load_result)

CTRL_2EP = [REPO_ROOT / "results" / "lumi_v060" / f"exp32_ctrl_s{s}"
            / "exp31_dense_control.json" for s in (42, 123, 456)]
CTRL_5EP = (REPO_ROOT / "results" / "lumi_v060" / "exp31_s42"
            / "exp31_dense_control.json")

# The wrong values are per-seed, because each is exp29's 5-epoch arm minus THAT
# seed's 2-epoch control: seed 42 reads -2.269 / -2.831 / -4.411, seed 123
# -2.117 / -2.679 / -4.259, seed 456 -2.055 / -2.617 / -4.197. Asserting the
# formula rather than the numbers is what makes this a defect test and not a
# transcription.
UNMATCHED_SEED42 = {"0.25": -2.269, "0.33": -2.831, "0.50": -4.411}
# The matched three-seed costs from EXP32_RATIO_AT_2EP.md that any table built
# from JSON must end up with instead.
MATCHED_2EP = {"0.25": -1.30, "0.33": -1.93, "0.50": -3.91}


# --------------------------------------------------------------------------- #
# the sidecar itself
# --------------------------------------------------------------------------- #

def test_sidecar_loads_and_validates():
    cs = load_corrections()
    assert cs, "CORRECTIONS.json has no corrections"
    ids = [c["id"] for c in cs]
    assert len(ids) == len(set(ids)), "duplicate correction ids"
    for c in cs:
        assert c["reason"].strip(), f"{c['id']} has an empty reason"
        assert c["provenance"].strip(), f"{c['id']} has no provenance"
        assert c["files"], f"{c['id']} names no files"


def test_every_correction_still_applies():
    """A stale correction means the run was redone and nobody checked."""
    bad = [r for r in audit() if r["status"] != "ok"]
    assert not bad, "corrections no longer applying: " + json.dumps(bad, indent=1)


def test_schema_version_is_enforced(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"schema_version": 99, "corrections": []}))
    with pytest.raises(CorrectionError, match="schema_version"):
        load_corrections(p)


@pytest.mark.parametrize("missing", ["id", "files", "pointer", "op", "reason",
                                     "provenance"])
def test_a_correction_without_provenance_is_rejected(tmp_path, missing):
    c = {"id": "x", "files": ["a.json"], "pointer": "a", "op": "remove",
         "reason": "r", "provenance": "p"}
    c.pop(missing)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"schema_version": 1, "corrections": [c]}))
    with pytest.raises(CorrectionError, match=missing):
        load_corrections(p)


def test_unsupported_expression_is_refused(tmp_path):
    """A data file must not become an expression evaluator."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"a": {"b": 1}}))
    c = [{"id": "e", "files": [str(p)], "pointer": "a", "op": "annotate",
          "value": {"x": "=__import__('os').system('echo hi')"},
          "reason": "r", "provenance": "p"}]
    with pytest.raises(CorrectionError, match="unsupported expression"):
        load_result(p, corrections=c)


# --------------------------------------------------------------------------- #
# the defect this exists for
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", CTRL_2EP, ids=lambda p: p.parent.name)
def test_disk_still_carries_the_defect(path):
    """Measured data is never mutated, so the bad values stay on disk.

    The defect is structural: each value is exp29's FIVE-epoch arm mean minus
    this file's TWO-epoch control. That formula is what the correction retires,
    so that is what the test pins.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    control = raw["control"]["accuracy"]
    for ratio, cell in raw["exp29_restated"]["ratios"].items():
        unmatched = (cell["exp29_mean"] - control) * 100
        assert cell["vs_this_control_pp"] == pytest.approx(unmatched, abs=1e-6), (
            "the committed file changed; if the run was redone, retire the "
            "correction instead of editing the data")
        # and it is genuinely wrong in one direction: it always overstates the
        # matched cost, by 0.29pp (P50, seed 456) to 0.76pp (P25, seed 456).
        assert cell["vs_this_control_pp"] < MATCHED_2EP[ratio] - 0.2


def test_seed_42_values_are_the_ones_quoted_in_the_sidecar():
    raw = json.loads(CTRL_2EP[0].read_text(encoding="utf-8"))
    got = {k: round(v["vs_this_control_pp"], 3)
           for k, v in raw["exp29_restated"]["ratios"].items()}
    assert got == UNMATCHED_SEED42


@pytest.mark.parametrize("path", CTRL_2EP, ids=lambda p: p.parent.name)
def test_corrected_read_drops_the_unmatched_cost(path):
    d = load_result(path)
    r = d["exp29_restated"]
    assert r["epochs_matched"] is False
    assert r["arm_epochs"] == 5 and r["control_epochs"] == 2
    for cell in r["ratios"].values():
        assert "vs_this_control_pp" not in cell, (
            "a 5-epoch arm against a 2-epoch control is not a pruning cost")
        # the fine-tuning gain side of the convention survives untouched
        assert "vs_pretrained_pp" in cell


@pytest.mark.parametrize("path", CTRL_2EP, ids=lambda p: p.parent.name)
def test_no_unmatched_value_survives_a_corrected_read(path):
    """Belt and braces: this seed's unmatched values may remain nowhere."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    control = raw["control"]["accuracy"]
    text = json.dumps(load_result(path))
    for cell in raw["exp29_restated"]["ratios"].values():
        assert f"{(cell['exp29_mean'] - control) * 100}" not in text


@pytest.mark.parametrize("path", CTRL_2EP, ids=lambda p: p.parent.name)
def test_fine_tuning_gain_is_computed_and_labelled(path):
    d = load_result(path)
    ctrl, pre = d["control"], d["pretrained"]
    assert ctrl["fine_tuning_gain_pp"] == pytest.approx(
        (ctrl["accuracy"] - pre["accuracy"]) * 100, abs=1e-9)
    assert ctrl["fine_tuning_gain_int4_pp"] == pytest.approx(
        (ctrl["int4_accuracy"] - pre["int4_accuracy"]) * 100, abs=1e-9)
    # it is the same quantity the old field held, under a name that says so
    assert ctrl["fine_tuning_gain_pp"] == pytest.approx(ctrl["vs_pretrained_pp"],
                                                        abs=1e-9)


def test_the_three_controls_agree_with_the_published_mean():
    """81.676 / 81.524 / 81.462 -> 81.554, the control every wave-1 cost uses."""
    accs = [load_result(p)["control"]["accuracy"] for p in CTRL_2EP]
    assert sorted(round(a * 100, 3) for a in accs) == [81.462, 81.524, 81.676]
    assert sum(accs) / 3 == pytest.approx(0.81554, abs=5e-6)


def test_matched_costs_are_recoverable_from_corrected_data():
    """The point of the exercise: a table built from JSON gets -1.30/-1.93/-3.91."""
    ctrl = sum(load_result(p)["control"]["accuracy"] for p in CTRL_2EP) / 3
    arms = {}
    for ratio in MATCHED_2EP:
        vals = []
        for s in (42, 123, 456):
            arm = json.loads((REPO_ROOT / "results" / "lumi_v060"
                              / f"exp32_ratios_s{s}" / "exp29_recovery.json"
                              ).read_text(encoding="utf-8"))
            vals.append(arm["ratios"][ratio]["recovered"]["accuracy"]
                        if "recovered" in arm["ratios"][ratio]
                        else arm["ratios"][ratio]["accuracy"])
        arms[ratio] = sum(vals) / 3
    for ratio, expected in MATCHED_2EP.items():
        assert (arms[ratio] - ctrl) * 100 == pytest.approx(expected, abs=0.02)


# --------------------------------------------------------------------------- #
# the matched control must not be swept up with the unmatched one
# --------------------------------------------------------------------------- #

def test_the_five_epoch_control_keeps_its_cost_but_is_flagged_single_seed():
    r = load_result(CTRL_5EP)["exp29_restated"]
    assert r["epochs_matched"] is True
    assert r["control_seeds"] == 1
    # matched, so the cost stands
    assert r["ratios"]["0.50"]["vs_this_control_pp"] == pytest.approx(-3.291,
                                                                      abs=0.01)
    assert "80.44" in r["preferred_control"]


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #

def test_load_result_does_not_write(tmp_path):
    src = json.loads(CTRL_2EP[0].read_text(encoding="utf-8"))
    before = CTRL_2EP[0].read_bytes()
    d = load_result(CTRL_2EP[0])
    d["control"]["accuracy"] = 0.0          # mutate the returned copy
    assert CTRL_2EP[0].read_bytes() == before
    assert json.loads(CTRL_2EP[0].read_text(encoding="utf-8")) == src


def test_corrections_applied_is_reported():
    d = load_result(CTRL_2EP[0])
    assert d["_corrections_applied"][0] == "exp32-ctrl-unmatched-restatement"
    assert d["_corrections_source"].endswith("CORRECTIONS.json")


def test_an_uncorrected_file_is_returned_clean():
    other = REPO_ROOT / "results" / "lumi_v060" / "exp33_s42" / "exp33_sparse.json"
    d = load_result(other)
    assert "_corrections_applied" not in d
    assert d["experiment"] == "exp33_sparse_vs_structured"


def test_sidecar_path_is_where_the_docs_say():
    assert CORRECTIONS_PATH.exists()
    assert CORRECTIONS_PATH.name == "CORRECTIONS.json"
