"""A cell that does not say what it is must not be scored.

The ViT gate cell and a ViT arm cell have the same keys, the same
``last_stage``, the same ``strategy`` and the same file name; only the size of
the numbers differs. Handed the whole results directory, the scorer folded the
4-minute gate run into G1 as a fourth seed and reported the band's mean as
72.2255 against the 78.6007 its three arms actually measured. So the runner
writes the role and the scorer refuses to guess one.

Every test here RUNS the scorer (rule 11): the refusal, the exclusion, and the
name collision that makes the declaration key a directory plus a file name.
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
from v070_gate_bands import CELL_ROLES, SCOREABLE_ROLE  # noqa: E402


def _cell(tmp_path, dirname, name, *, role=None, acc=None, cost=None,
          strategy="random", seed=42):
    d = {"strategy": strategy, "seed": seed, "mode": "paper",
         "last_stage": "validate", "recovered_accuracy": acc,
         "pruning_cost_pp": cost,
         "provenance": {"matches": True,
                        "recorded": {"commit": "deadbeef", "dirty": False},
                        "computed": {"aggregate_sha256": "abc"}}}
    if role is not None:
        d["role"] = role
    p = tmp_path / dirname / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_a_cell_without_a_role_stops_the_scoring(tmp_path, capsys):
    """Not dropped, not scored: dropping it silently is another guess."""
    paths = [str(_cell(tmp_path, "arm_s%d" % s, "gate_random_paper_s%d.json" % s,
                       acc=0.786, cost=-2.7, seed=s))
             for s in (42, 123, 456)]
    rc = S.main(["--target", "HEAD"] + paths)
    out = capsys.readouterr().out
    assert rc == 1
    assert "do not say whether they are an arm" in out
    assert "G1-paper" not in out, "nothing may be scored while a cell is unknown"


def test_the_gate_cell_is_excluded_and_the_arms_score_alone(tmp_path, capsys):
    """The exact failure, reproduced: three arms plus the gate cell."""
    arms = [str(_cell(tmp_path, "arm_s%d" % s, "gate_random_paper_s%d.json" % s,
                      role="arm", acc=acc, cost=-2.7, seed=s))
            for s, acc in ((42, 0.78606), (123, 0.78404), (456, 0.78792))]
    gate = str(_cell(tmp_path, "vit_gate", "gate_random_paper_s42.json",
                     role="gate", acc=0.531, cost=-28.576))
    S.main(["--target", "HEAD"] + arms + [gate])
    out = capsys.readouterr().out
    assert "arms scored: 3" in out and "not arms: 1" in out
    assert "n=3" in out
    assert "72.2" not in out, "the gate cell is back inside the band"
    assert "78.6007" in out


def test_the_declaration_is_keyed_by_directory_because_names_collide(tmp_path):
    """Both files are named gate_random_paper_s42.json. That is the trap."""
    arm = _cell(tmp_path, "vit_random_s42", "gate_random_paper_s42.json",
                acc=0.78606, cost=-2.7)
    gate = _cell(tmp_path, "vit_gate", "gate_random_paper_s42.json",
                 acc=0.531, cost=-28.576)
    assert arm.name == gate.name
    decl = tmp_path / "roles.json"
    decl.write_text(json.dumps({"cells": {
        "vit_random_s42/gate_random_paper_s42.json": {"role": "arm",
                                                      "why": "job 1"},
        "vit_gate/gate_random_paper_s42.json": {"role": "gate",
                                                "why": "job 2"}}}),
        encoding="utf-8")
    declared = S.declared_roles(str(decl))
    cells = S.load_cells([str(arm), str(gate)])
    keep, skipped, unknown = S.resolve_roles(cells, declared)
    assert not unknown
    assert [c["_role"] for c in keep] == ["arm"]
    assert [c["_role"] for c in skipped] == ["gate"]
    assert "job 1" in keep[0]["_role_source"]


def test_a_role_in_the_file_beats_the_declaration(tmp_path):
    arm = _cell(tmp_path, "d", "c.json", role="gate", acc=0.5, cost=-1.0)
    decl = tmp_path / "roles.json"
    decl.write_text(json.dumps({"cells": {"d/c.json": {"role": "arm"}}}),
                    encoding="utf-8")
    keep, skipped, unknown = S.resolve_roles(
        S.load_cells([str(arm)]), S.declared_roles(str(decl)))
    assert not keep and not unknown
    assert skipped[0]["_role_source"] == "the result file"


def test_an_unknown_role_raises_rather_than_being_treated_as_an_arm(tmp_path):
    arm = _cell(tmp_path, "d", "c.json", role="smoke", acc=0.5, cost=-1.0)
    with pytest.raises(ValueError, match="smoke"):
        S.resolve_roles(S.load_cells([str(arm)]), {})
    decl = tmp_path / "roles.json"
    decl.write_text(json.dumps({"cells": {"d/c.json": {"role": "smoke"}}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="smoke"):
        S.declared_roles(str(decl))


def test_only_the_arm_role_is_scoreable():
    assert SCOREABLE_ROLE == "arm" and SCOREABLE_ROLE in CELL_ROLES
    assert set(CELL_ROLES) == {"arm", "gate", "micro"}


def test_the_runner_refuses_a_role_it_does_not_know():
    """The other end of the contract: run_cell will not record a wrong role."""
    import gate_v070_vit as G
    with pytest.raises(ValueError, match="role must be one of"):
        G.run_cell(None, None, None, strategy="random", seed=42, pruning=0.5,
                   epochs=1, mode="paper", control=None, outdir=".",
                   label="x", amp=None, role="smoke")
