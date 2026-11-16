# -*- coding: utf-8 -*-
"""Controls are found by what they are, and belong to one model.

Two defects these tests pin, one of which cost sixteen jobs.

``CONTROL_SOURCES`` was a fixed list of three seeds and three paths,
written when those were the only seeds. Wave 3's new-seed controls landed
in directories the registry had never heard of, so every arm at a new seed
died with "no 2-epoch control for seed 7" while the control sat on disk
beside it, and no job dependency could have helped.

The ladder also had no concept of a model, so an OPT arm asking for a
2-epoch control would have been handed ViT-B/16's 81.554% without a word.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import matched_control as M  # noqa: E402

VIT = "google/vit-base-patch16-224"
OPT = "facebook/opt-6.7b"


def _write(root: Path, rel: str, *, model: str, epochs: int, seed: int,
           acc: float, int4: float | None = None, heads_removed: int = 0):
    fp = root / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({
        "experiment": "exp31_dense_control", "model": model,
        "seed": seed, "epochs": epochs, "heads_removed": heads_removed,
        "control": {"accuracy": acc, "int4_accuracy": int4,
                    "epochs": epochs},
    }), encoding="utf-8")
    return fp


def test_a_new_seed_is_discovered(tmp_path):
    """The exact failure: a control that exists and could not be seen."""
    _write(tmp_path, "results/w3_ctrl2_s7/exp31_dense_control.json",
           model=VIT, epochs=2, seed=7, acc=0.8161)
    found = M.discover_controls(tmp_path, model=VIT)
    assert [(f["epochs"], f["seed"]) for f in found] == [(2, 7)]
    assert found[0]["accuracy"] == 0.8161


def test_discovery_reads_the_file_not_the_path(tmp_path):
    """A run written to an unexpected directory is still found."""
    _write(tmp_path, "results/somewhere_unexpected/exp31_dense_control.json",
           model=VIT, epochs=2, seed=99, acc=0.80)
    found = M.discover_controls(tmp_path, model=VIT)
    assert [(f["epochs"], f["seed"]) for f in found] == [(2, 99)]


def test_a_pruned_run_is_not_a_control(tmp_path):
    """`heads_removed` makes it an arm, whatever the filename says."""
    _write(tmp_path, "results/w3_ctrl2_s8/exp31_dense_control.json",
           model=VIT, epochs=2, seed=8, acc=0.77, heads_removed=72)
    assert M.discover_controls(tmp_path, model=VIT) == []


def test_another_model_is_not_returned(tmp_path):
    """An OPT control must never surface in a ViT query."""
    _write(tmp_path, "results/w3_ctrl2_s7/exp31_dense_control.json",
           model=OPT, epochs=2, seed=7, acc=0.50)
    assert M.discover_controls(tmp_path, model=VIT) == []
    got = M.discover_controls(tmp_path, model=OPT)
    assert len(got) == 1 and got[0]["model"] == OPT


def test_both_models_are_kept_apart(tmp_path):
    """The case that must never silently work."""
    _write(tmp_path, "results/w3_ctrl2_s7/exp31_dense_control.json",
           model=VIT, epochs=2, seed=7, acc=0.8161)
    _write(tmp_path, "results/opt67_ctrl_s7/exp31_dense_control.json",
           model=OPT, epochs=2, seed=7, acc=0.50)
    vit = M.discover_controls(tmp_path, model=VIT)
    opt = M.discover_controls(tmp_path, model=OPT)
    assert len(vit) == 1 and len(opt) == 1
    assert vit[0]["accuracy"] != opt[0]["accuracy"]


def test_the_ladder_refuses_another_models_arm():
    """A 6.7B arm may not be scored against a 1.3B or ViT control."""
    ladder = M.ControlLadder.load((2,), verify=True)
    assert ladder.model == M.DEFAULT_MODEL
    ladder.at(2)                       # same model: fine
    with pytest.raises(M.ControlError) as e:
        ladder.at(2, for_model=OPT)
    msg = str(e.value)
    assert OPT in msg and M.DEFAULT_MODEL in msg, msg
    assert "category error" in msg, msg


def test_discovery_does_not_move_the_verified_values():
    """Finding more controls must not change what was published.

    Discovery widens what can be found; `verify()` still checks the
    registered three-seed means against the published figures, and those
    must read exactly as before.
    """
    ladder = M.ControlLadder.load((2, 5), verify=True)
    assert ladder.at(2).seeds == 3
    assert abs(ladder.at(2).accuracy - 0.81554) < 1e-9


def test_a_second_file_does_not_block_the_load(tmp_path, monkeypatch):
    """Superseded behaviour, kept as a test so the change is explicit.

    An earlier version raised ControlError when two files described the
    same (model, budget, seed). That fired on its first real encounter --
    wave 3's re-measured controls -- and blocked every arm needing a cost,
    for what is a legitimate repeat. Loading must succeed; what the repeat
    is worth is the next test's business.
    """
    a = _write(tmp_path, "results/lumi_v060/exp32_ctrl_s42/"
                         "exp31_dense_control.json",
               model=VIT, epochs=2, seed=42, acc=0.81676, int4=0.80378)
    _write(tmp_path, "results/w3_ctrl2_s42/exp31_dense_control.json",
           model=VIT, epochs=2, seed=42, acc=0.79999)
    monkeypatch.setattr(M, "CONTROL_SOURCES",
                        {2: [str(a.relative_to(tmp_path)).replace("\\", "/")]})
    monkeypatch.setattr(M, "PUBLISHED", {})
    ladder = M.ControlLadder.load((2,), verify=False, root=tmp_path)
    assert ladder.at(2, seed=42).accuracy == 0.81676


def test_a_repeated_control_is_recorded_not_refused(tmp_path, monkeypatch):
    """Wave 3 re-measured the 2-epoch controls for seeds 42/123/456.

    The first version of this guard read that as two files contradicting
    each other and refused to load at all, which blocked every arm that
    needed a cost. A second measurement of the same (model, budget, seed)
    is a repeat: the registered value stays authoritative and the repeat
    is kept beside it, because the run-to-run variance of a control is
    worth knowing and is otherwise invisible.
    """
    a = _write(tmp_path, "results/lumi_v060/exp32_ctrl_s42/"
                         "exp31_dense_control.json",
               model=VIT, epochs=2, seed=42, acc=0.81676, int4=0.80378)
    _write(tmp_path, "results/w3_ctrl2_s42/exp31_dense_control.json",
           model=VIT, epochs=2, seed=42, acc=0.81590)
    monkeypatch.setattr(M, "CONTROL_SOURCES",
                        {2: [str(a.relative_to(tmp_path)).replace("\\", "/")]})
    monkeypatch.setattr(M, "PUBLISHED", {})

    ladder = M.ControlLadder.load((2,), verify=False, root=tmp_path)

    # The registered value is the one used.
    assert ladder.at(2, seed=42).accuracy == 0.81676
    assert "exp32_ctrl_s42" in ladder.at(2, seed=42).source

    # And the repeat is on the record, with its delta.
    assert len(ladder.repeats) == 1
    rep = ladder.repeats[0]
    assert rep["epochs"] == 2 and rep["seed"] == 42
    assert "w3_ctrl2_s42" in rep["repeat"]
    assert abs(rep["delta_pp"] - (0.81590 - 0.81676) * 100) < 1e-9


def test_an_identical_repeat_is_not_even_noted(tmp_path, monkeypatch):
    """Nothing to report when the two agree exactly."""
    a = _write(tmp_path, "results/lumi_v060/exp32_ctrl_s42/"
                         "exp31_dense_control.json",
               model=VIT, epochs=2, seed=42, acc=0.81676)
    _write(tmp_path, "results/w3_ctrl2_s42/exp31_dense_control.json",
           model=VIT, epochs=2, seed=42, acc=0.81676)
    monkeypatch.setattr(M, "CONTROL_SOURCES",
                        {2: [str(a.relative_to(tmp_path)).replace("\\", "/")]})
    monkeypatch.setattr(M, "PUBLISHED", {})
    ladder = M.ControlLadder.load((2,), verify=False, root=tmp_path)
    assert ladder.repeats == []
