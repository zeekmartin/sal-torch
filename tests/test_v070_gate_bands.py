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


# ----------------------------------------------------------------- provenance
def _staged(tmp_path, commit="0" * 40, dirty=False, tamper=False):
    """A staged package plus its sidecar, exactly as the cluster holds one."""
    import json
    import types

    tmp_path.mkdir(parents=True, exist_ok=True)
    pkg = tmp_path / "sal"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '0.6.1'\n", encoding="utf-8")
    (pkg / "recovery.py").write_text("def select_precision(): pass\n",
                                     encoding="utf-8")
    from v070_gate_bands import package_manifest
    man = package_manifest(pkg)
    (tmp_path / "PROVENANCE.json").write_text(json.dumps({
        "commit": commit, "dirty": dirty,
        "n_files": man["n_files"],
        "aggregate_sha256": man["aggregate_sha256"],
        "contains_precision_change": True,
        "recorded_at": "2026-11-20T13:00:00+00:00"}), encoding="utf-8")
    if tamper:
        # The package changed after the sidecar was written, which is the
        # thing the comparison exists to catch.
        (pkg / "recovery.py").write_text("def select_precision(): return 1\n",
                                         encoding="utf-8")
    mod = types.SimpleNamespace(__file__=str(pkg / "__init__.py"),
                                __version__="0.6.1")
    return mod


def test_the_manifest_is_order_independent_and_content_sensitive(tmp_path):
    from v070_gate_bands import package_manifest

    d = tmp_path / "pkg"
    d.mkdir()
    (d / "b.py").write_text("b\n", encoding="utf-8")
    (d / "a.py").write_text("a\n", encoding="utf-8")
    first = package_manifest(d)
    assert first["n_files"] == 2 and len(first["files"]) == 2
    assert package_manifest(d)["aggregate_sha256"] == first["aggregate_sha256"]
    (d / "a.py").write_text("a!\n", encoding="utf-8")
    assert package_manifest(d)["aggregate_sha256"] != first["aggregate_sha256"]


def test_provenance_matches_an_untouched_package(tmp_path):
    from v070_gate_bands import package_provenance

    prov = package_provenance(_staged(tmp_path))
    assert prov["matches"] is True
    assert prov["recorded"]["contains_precision_change"] is True
    assert prov["computed"]["n_files"] == 2


def test_provenance_catches_a_package_edited_after_recording(tmp_path):
    from v070_gate_bands import package_provenance

    prov = package_provenance(_staged(tmp_path, tamper=True))
    assert prov["matches"] is False
    assert "NOT what the sidecar records" in prov["why"]


def test_provenance_is_absent_without_a_sidecar(tmp_path):
    import types

    from v070_gate_bands import package_provenance

    pkg = tmp_path / "sal"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    mod = types.SimpleNamespace(__file__=str(pkg / "__init__.py"))
    prov = package_provenance(mod)
    assert prov["recorded"] is None and prov["matches"] is False
    assert "cannot say which commit" in prov["why"]


def test_verify_provenance_refuses_everything_it_should(tmp_path):
    import subprocess

    from v070_gate_bands import package_provenance, verify_provenance

    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                          capture_output=True, text=True).stdout.strip()

    assert not verify_provenance(None)["ok"]
    assert "no provenance" in verify_provenance(None)["why"]

    tampered = package_provenance(_staged(tmp_path / "t", tamper=True))
    assert not verify_provenance(tampered, repo=repo)["ok"]

    dirty = package_provenance(_staged(tmp_path / "d", commit=head, dirty=True))
    v = verify_provenance(dirty, repo=repo)
    assert not v["ok"] and "dirty" in v["why"]

    unknown = package_provenance(_staged(tmp_path / "u", commit="1" * 40))
    v = verify_provenance(unknown, repo=repo)
    assert not v["ok"] and "not an ancestor" in v["why"]

    good = package_provenance(_staged(tmp_path / "g", commit=head))
    v = verify_provenance(good, repo=repo, target="HEAD")
    assert v["ok"] and v["commit"] == head


def test_a_band_with_perfect_numbers_fails_on_bad_provenance(tmp_path):
    """The numbers and the code are two claims, and both have to hold."""
    from v070_gate_bands import package_provenance

    results = {n: {"per_seed": v[0], "cost": v[1]}
               for n, v in ARCHIVED.items()}
    repo = Path(__file__).resolve().parents[1]

    bad = package_provenance(_staged(tmp_path, commit="1" * 40))
    out = check_all(results, provenance=bad, repo=repo)
    assert out["passed"] is False
    assert all(not v["passed"] for v in out["bands"].values())
    assert any("provenance" in w for w in
               out["bands"]["G1-paper"]["why"])
    assert out["provenance"]["ok"] is False

    # Without provenance the scorer still works, and says it checked none.
    plain = check_all(results)
    assert plain["passed"] is True and plain["provenance"] is None


# ------------------------------------------------------------- the scorer
def _cell(strategy, seed, value, cost, prov, stage="validate", ppl=False):
    c = {"strategy": strategy if not ppl else None,
         "mode": strategy if ppl else "paper",
         "seed": seed, "last_stage": stage, "provenance": prov}
    if ppl:
        c["recovered_ppl"] = value
        c["pruning_cost"] = cost
    else:
        c["recovered_accuracy"] = value / 100
        c["pruning_cost_pp"] = cost
    return c


def test_the_scorer_groups_cells_into_the_bands_they_belong_to(tmp_path):
    import score_v070_gate as S
    from v070_gate_bands import package_provenance

    prov = package_provenance(_staged(tmp_path))
    cells = [_cell("random", s, v, -2.738, prov)
             for s, v in zip((42, 123, 456), ARCHIVED["G1-paper"][0])]
    scored = S.group(cells)
    assert set(scored) == {"G1-paper", "G1-default"}
    assert scored["G1-paper"]["per_seed"] == list(ARCHIVED["G1-paper"][0])
    assert scored["G1-paper"]["cost"] == pytest.approx(-2.738)


def test_a_cell_that_never_reached_validate_is_not_scored(tmp_path):
    import score_v070_gate as S
    from v070_gate_bands import package_provenance

    prov = package_provenance(_staged(tmp_path))
    cells = [_cell("random", 42, 78.838, -2.738, prov),
             _cell("random", 123, 78.668, -2.738, prov, stage="recover"),
             _cell("random", 456, 78.942, -2.738, prov)]
    scored = S.group(cells)
    assert scored["G1-paper"]["per_seed"] == [78.838, 78.942], (
        "a run that stopped before validate has no scoreable number, and "
        "averaging over whatever finished turns two seeds into three")
    assert scored["G1-paper"]["n_cells"] == 2


def test_a_missing_cost_leaves_the_cost_absent(tmp_path):
    import score_v070_gate as S
    from v070_gate_bands import package_provenance

    prov = package_provenance(_staged(tmp_path))
    cells = [_cell("random", 42, 78.838, -2.738, prov),
             _cell("random", 123, 78.668, None, prov),
             _cell("random", 456, 78.942, -2.738, prov)]
    assert S.group(cells)["G1-paper"]["cost"] is None, (
        "a mean over the two seeds that reported one is not the arm's cost")


def test_provenance_is_enforced_per_band_not_per_batch(tmp_path):
    """Three seeds of one band must be one package; two bands may differ.

    The ViT and OPT families of the real gate ran a parent commit and its
    child, because a bug found by the first ViT arms was fixed between them.
    Demanding one package across the whole batch would refuse that, although
    each family is internally consistent and both are in the history being
    published.

    This needs two real commits, so it needs real history. A shallow clone
    has none: `actions/checkout` defaults to `fetch-depth: 1`, `HEAD~1` does
    not resolve, and this test used to discard the exit status of its own
    `git rev-parse` and compare an empty string, which failed as a bare
    `assert False` with nothing saying why. Both workflows now check out the
    full history so the test runs for real; the skip is for anyone running
    the suite from an archive or a depth-limited clone, and it says so.
    """
    import subprocess

    import score_v070_gate as S
    from v070_gate_bands import package_provenance

    repo = Path(__file__).resolve().parents[1]

    def rev(ref):
        out = subprocess.run(["git", "rev-parse", ref], cwd=str(repo),
                             capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.strip():
            pytest.skip("no %s in this checkout, so there is no second "
                        "commit to build a different package from: this "
                        "needs real git history, not a shallow clone" % ref)
        return out.stdout.strip()

    head = rev("HEAD")
    parent = rev("HEAD~1")

    older = package_provenance(_staged(tmp_path / "a", commit=parent))
    newer = package_provenance(_staged(tmp_path / "b", commit=head))

    # One band, one package: fine.
    one = {"n_cells": 3, "provenances": [newer, newer, newer]}
    v = S.band_provenance(one, "HEAD", repo)
    assert v["ok"], v["why"]

    # Another band on the parent commit: also fine, it is an ancestor.
    other = {"n_cells": 3, "provenances": [older, older, older]}
    assert S.band_provenance(other, "HEAD", repo)["ok"]

    # The same band split across two packages: refused.
    mixed = {"n_cells": 3, "provenances": [older, newer, newer]}
    v = S.band_provenance(mixed, "HEAD", repo)
    assert not v["ok"] and "different packages" in v["why"]

    # A cell with no provenance at all: refused, and counted.
    missing = {"n_cells": 3, "provenances": [newer, newer, None]}
    v = S.band_provenance(missing, "HEAD", repo)
    assert not v["ok"] and "carry no provenance" in v["why"]
