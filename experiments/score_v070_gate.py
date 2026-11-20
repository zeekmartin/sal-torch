#!/usr/bin/env python3
"""Score the release gate, and refuse a band whose code would not be pushed.

A band is two claims, not one: the numbers landed inside the pre-registered
band, **and** they were produced by code that is in the history being
published. The second half cannot be checked on the cluster, because the
scratch checkout is not a git repository, so it is checked here against the
provenance each cell recorded: the commit the staged package was built from,
and a SHA-256 over its `sal/*.py` files which the job itself recomputed and
compared.

What this refuses, and why each one matters:

* **a cell that does not say what it is.** A gate cell, a micro cell and an
  arm cell have the same keys, the same ``last_stage`` and the same shape, and
  the only thing separating them was how large their numbers were. Passed the
  whole results directory, this scorer folded the 4-minute ViT gate cell into
  G1 as a fourth seed and reported the band's mean as 72.2255 instead of
  78.6007. The role is now **written by the runner**, never inferred from a
  step count or a file name, and a cell without one stops the scoring rather
  than being quietly dropped: dropping it would be another guess.

* **no provenance in the result.** Then nothing says which code ran, and a
  gate that cannot say that proves nothing about what ships.
* **a package that does not match its sidecar.** Someone edited the staged
  copy after it was recorded, so the recorded commit describes something else.
* **a staged copy built from a dirty working tree.** Then no commit describes
  what ran.
* **a commit that is not an ancestor of the push target.** The result came
  from code that is not in what would be pushed, so it says nothing about it.

Usage::

    python experiments/score_v070_gate.py results/gate_v070_*/gate_*.json
    python experiments/score_v070_gate.py --target HEAD --json <files...>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Where the archived decoder run lives in the research repository, relative
#: to its root. Passed explicitly in practice; this is the documented default.
DEFAULT_FLOOR_ARCHIVE = ("results/exp36_rec_heads_random_50/"
                         "exp36_opt_sparse.json")
sys.path.insert(0, str(HERE))

from v070_gate_bands import (BANDS, BANDS_BY_CELL, CELL_ROLES,  # noqa: E402
                             EXPLAINED_BY_NOISE, SCOREABLE_ROLE, check_all,
                             explain_by_noise, measured_floor,
                             verify_provenance)

#: The archived decoder cell the floor is measured against, and the one-shot
#: perplexity that proves a re-run removed the same heads. A re-run whose
#: one-shot differs is measuring a different model and its floor is void.
ARCHIVE_SEED = 42
ARCHIVE_ONE_SHOT = 93.5936


def decoder_floor(rerun_path, archive_path, seed: int = ARCHIVE_SEED) -> tuple:
    """``(floor, source)``: ``|re-run - archive|`` for one seed of one cell.

    Both numbers are READ, never typed, and the re-run is void unless its
    one-shot perplexity matches the archive's to four decimals: that is what
    proves the same heads were removed, and without it the two runs differ by
    more than the run.
    """
    def arm(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d["arms"]["heads_random_50"]["seeds"][str(seed)]

    a, b = arm(archive_path), arm(rerun_path)
    for key in ("oneshot_wikitext2_ppl", "recovered_wikitext2_ppl"):
        for src, where in ((a, archive_path), (b, rerun_path)):
            if key not in src:
                raise KeyError("%s has no %s for seed %d" % (where, key, seed))
    if round(b["oneshot_wikitext2_ppl"], 4) != round(a["oneshot_wikitext2_ppl"], 4):
        raise ValueError(
            "the re-run's one-shot perplexity is %.4f and the archive's is "
            "%.4f, so it did not remove the same heads and its spread is not "
            "a same-seed floor"
            % (b["oneshot_wikitext2_ppl"], a["oneshot_wikitext2_ppl"]))
    floor = measured_floor(b["recovered_wikitext2_ppl"],
                           a["recovered_wikitext2_ppl"])
    return floor, ("seed %d re-run %.4f against the archive's %.4f"
                   % (seed, b["recovered_wikitext2_ppl"],
                      a["recovered_wikitext2_ppl"]))


def _cell_key(path) -> str:
    """``"<parent directory>/<file name>"``, which is unique where a name is not."""
    q = Path(path)
    return "%s/%s" % (q.parent.name, q.name)


def declared_roles(path) -> dict:
    """``{file basename: {"role", "why"}}`` from an explicit declaration.

    Cells written before the runner recorded its own role cannot be given one
    retroactively by editing them: measured data is never mutated. They get
    one here instead, in a file that states for each cell WHY the role is
    known, so the claim is auditable rather than assumed. A cell whose own
    JSON carries a role ignores this file.

    Keys are ``"<directory>/<file>"``, not the file name alone, because the
    names collide by design: the ViT gate job and the seed-42 arm both write
    ``gate_random_paper_s42.json``, and keying on the name would hand the gate
    cell the arm's declaration. That collision is the same one this whole
    mechanism exists to stop, so it is not papered over here.
    """
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for key, entry in d.get("cells", {}).items():
        role = entry.get("role") if isinstance(entry, dict) else entry
        if role not in CELL_ROLES:
            raise ValueError(
                "the role declaration gives %r the role %r, which is not one "
                "of %s" % (key, role, CELL_ROLES))
        out[_cell_key(key)] = {"role": role,
                               "why": (entry.get("why", "")
                                       if isinstance(entry, dict) else "")}
    return out


def resolve_roles(cells, declared: dict) -> tuple:
    """``(scoreable cells, skipped, unknown)``, with nothing inferred.

    A cell's own ``role`` wins. Failing that, the declaration may supply one.
    Failing both, the cell is **unknown** and the caller stops: a cell nobody
    can classify is a cell nobody knows the meaning of, and both folding it
    into a band and dropping it are guesses.
    """
    keep, skipped, unknown = [], [], []
    for c in cells:
        name = _cell_key(c["_path"])
        role, source = c.get("role"), "the result file"
        if role is None and name in declared:
            role, source = declared[name]["role"], (
                "declared: " + (declared[name]["why"] or "no reason given"))
        if role is None:
            unknown.append(c)
            continue
        if role not in CELL_ROLES:
            raise ValueError("%s carries the role %r, which is not one of %s"
                             % (name, role, CELL_ROLES))
        entry = dict(c, _role=role, _role_source=source)
        (keep if role == SCOREABLE_ROLE else skipped).append(entry)
    return keep, skipped, unknown


def load_cells(paths) -> list:
    """Every per-cell result file, with the ones that never finished dropped."""
    cells = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        cells.append(dict(d, _path=str(p)))
    return cells


def group(cells) -> dict:
    """``{band name: {"per_seed": [...], "cost": mean}}`` from the cells.

    A cell contributes only if it reached `validate`: a run that stopped at
    `recover` has no scoreable number, and silently averaging over whatever
    finished is how a two-seed result gets reported as a three-seed one.
    """
    by_key: dict = {}
    for c in cells:
        key = c.get("strategy") or c.get("mode")
        if c.get("last_stage") != "validate":
            continue
        acc = c.get("recovered_accuracy")
        ppl = c.get("recovered_ppl")
        value = (acc * 100 if acc is not None else ppl)
        if value is None:
            continue
        cost = (c.get("pruning_cost_pp") if acc is not None
                else c.get("pruning_cost"))
        by_key.setdefault(key, {"values": [], "costs": [], "cells": [],
                                "seeds": []})
        by_key[key]["values"].append(value)
        by_key[key]["costs"].append(cost)
        by_key[key]["cells"].append(c)
        # The seed LABEL travels with the value, because the pre-registered
        # noise rule compares each seed against its own archived seed and the
        # cells arrive in file-system order.
        by_key[key]["seeds"].append(c.get("seed"))
    out = {}
    for key, v in by_key.items():
        costs = [c for c in v["costs"] if c is not None]
        mean_cost = (sum(costs) / len(costs)
                     if len(costs) == len(v["values"]) and costs else None)
        provs = [c.get("provenance") for c in v["cells"]]
        for band in BANDS_BY_CELL.get(key, ()):
            out[band] = {"per_seed": v["values"], "cost": mean_cost,
                         "seeds": v["seeds"],
                         "n_cells": len(v["cells"]), "provenances": provs}
    return out


def band_provenance(entry: dict, target: str, repo) -> dict:
    """The provenance verdict for ONE band, over its own cells.

    Per band rather than per batch, and the distinction is not cosmetic: the
    three seeds of a band have to be the same code, or they are not one
    measurement, while two different bands may legitimately have run different
    commits as long as each is in the history being published. Enforcing one
    hash across the whole batch would refuse a perfectly sound gate whose
    families ran a parent and its child.
    """
    provs = [p for p in entry.get("provenances", []) if p]
    if len(provs) != entry.get("n_cells"):
        return {"ok": False,
                "why": "%d of %d cells carry no provenance"
                       % (entry.get("n_cells", 0) - len(provs),
                          entry.get("n_cells", 0))}
    hashes = {p.get("computed", {}).get("aggregate_sha256") for p in provs}
    commits = {(p.get("recorded") or {}).get("commit") for p in provs}
    if len(hashes) > 1 or len(commits) > 1:
        return {"ok": False,
                "why": "this band's %d cells ran %d different packages, so "
                       "they are not one measurement"
                       % (len(provs), max(len(hashes), len(commits)))}
    v = verify_provenance(provs[0], target=target, repo=repo)
    v["n_cells"] = len(provs)
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--target", default="HEAD",
                   help="what would be pushed; the recorded commit must be an "
                        "ancestor of it")
    p.add_argument("--decoder-floor-rerun", default=None,
                   help="the re-run of the archived decoder script whose "
                        "spread against the archive is the measured same-seed "
                        "floor (job 23898426's exp36_opt_sparse.json)")
    p.add_argument("--decoder-floor-archive", default=None,
                   help="the archived run it is compared against")
    p.add_argument("--roles", default=None,
                   help="a JSON declaration giving the role of cells whose "
                        "own result file predates the runner recording it")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    all_cells = load_cells(args.files)
    cells, skipped, unknown = resolve_roles(
        all_cells, declared_roles(args.roles))
    if unknown:
        print("NOT SCORED: %d of %d cells do not say whether they are an arm, "
              "a gate or a micro run, and this scorer does not guess:"
              % (len(unknown), len(all_cells)))
        for c in unknown:
            print("  %s" % c["_path"])
        print("Re-run them with a runner that records `role`, or declare them "
              "with --roles.")
        return 1
    scored = group(cells)

    # Numbers first, then each band's own provenance. A band passes only if
    # both hold.
    verdict = check_all(scored)
    per_band = {}
    for name, entry in scored.items():
        pv = band_provenance(entry, args.target, HERE.parent)
        per_band[name] = pv
        if not pv["ok"]:
            verdict["bands"][name]["passed"] = False
            verdict["bands"][name]["why"] = list(
                verdict["bands"][name]["why"]) + ["provenance: " + pv["why"]]
    # The pre-registered noise rule, applied after provenance so that a band
    # failing on provenance can never be explained away as noise.
    floor, floor_source = None, ""
    if args.decoder_floor_rerun:
        floor, floor_source = decoder_floor(
            args.decoder_floor_rerun,
            args.decoder_floor_archive or DEFAULT_FLOOR_ARCHIVE)
    for name, v in list(verdict["bands"].items()):
        band = BANDS.get(name)
        if band is not None:
            verdict["bands"][name] = explain_by_noise(v, band, floor,
                                                      floor_source)
    verdict["passed"] = all(v["passed"] for v in verdict["bands"].values())
    verdict["blocking"] = [n for n, v in verdict["bands"].items()
                           if v.get("blocking", not v["passed"])]
    verdict["measured_floor"] = floor
    verdict["provenance"] = per_band

    if args.json:
        print(json.dumps({"verdict": verdict,
                          "roles": {_cell_key(c["_path"]):
                                    {"role": c["_role"],
                                     "source": c["_role_source"]}
                                    for c in cells + skipped},
                          "scored": {k: {kk: vv for kk, vv in v.items()
                                         if kk != "provenances"}
                                     for k, v in scored.items()},
                          "n_cells": len(cells)}, indent=1, default=str))
        return 0 if (verdict["passed"] and verdict["complete"]) else 1

    print("=== v0.7.0 gate, scored against the pre-registration ===")
    print("cells read: %d | arms scored: %d | not arms: %d | bands: %d of %d"
          % (len(all_cells), len(cells), len(skipped),
             verdict["n_checked"], verdict["n_bands"]))
    for c in skipped:
        print("  skipped %s: role %s (%s)"
              % (_cell_key(c["_path"]), c["_role"], c["_role_source"]))
    print()
    for name in BANDS:
        v = verdict["bands"].get(name)
        if v is None:
            print("  %-12s NOT SCORED (no cell produced it)" % name)
            continue
        print("  %-12s %s" % (name, v.get("status",
                                           "PASS" if v["passed"] else "FAIL")))
        print("      n=%d  mean %.4f  per-seed %s  cost %s"
              % (v["n_seeds"], v["mean"],
                 ["%.4f" % x for x in v["per_seed"]],
                 "none" if v["cost"] is None else "%.4f" % v["cost"]))
        pv = per_band.get(name)
        if pv:
            print("      code: %s" % pv["why"])
        for why in v["why"]:
            print("      %s" % why)
    print()
    complete = verdict["complete"]
    if floor is not None:
        print("measured same-seed floor: %.4f (%s)" % (floor, floor_source))
    blocking = verdict["blocking"]
    if verdict["passed"] and complete:
        print("ALL SIX BANDS PASS")
    elif complete and not blocking:
        print("NOT ALL SIX PASS, AND NOTHING BLOCKS THE TAG: %s"
              % ", ".join("%s is %s" % (n, verdict["bands"][n]["status"])
                          for n, v in verdict["bands"].items()
                          if not v["passed"]))
    else:
        print("NOT A PASS: %s" % (("these bands block the tag: %s"
                                   % ", ".join(blocking)) if complete
                                  else "not every band was scored"))
    return 0 if (complete and not blocking) else 1


if __name__ == "__main__":
    raise SystemExit(main())
