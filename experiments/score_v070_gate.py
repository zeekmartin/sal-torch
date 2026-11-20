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
sys.path.insert(0, str(HERE))

from v070_gate_bands import (BANDS, BANDS_BY_CELL, check_all,  # noqa: E402
                             verify_provenance)


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
        by_key.setdefault(key, {"values": [], "costs": [], "cells": []})
        by_key[key]["values"].append(value)
        by_key[key]["costs"].append(cost)
        by_key[key]["cells"].append(c)
    out = {}
    for key, v in by_key.items():
        costs = [c for c in v["costs"] if c is not None]
        mean_cost = (sum(costs) / len(costs)
                     if len(costs) == len(v["values"]) and costs else None)
        provs = [c.get("provenance") for c in v["cells"]]
        for band in BANDS_BY_CELL.get(key, ()):
            out[band] = {"per_seed": v["values"], "cost": mean_cost,
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
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cells = load_cells(args.files)
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
    verdict["passed"] = all(v["passed"] for v in verdict["bands"].values())
    verdict["provenance"] = per_band

    if args.json:
        print(json.dumps({"verdict": verdict,
                          "scored": {k: {kk: vv for kk, vv in v.items()
                                         if kk != "provenances"}
                                     for k, v in scored.items()},
                          "n_cells": len(cells)}, indent=1, default=str))
        return 0 if (verdict["passed"] and verdict["complete"]) else 1

    print("=== v0.7.0 gate, scored against the pre-registration ===")
    print("cells read: %d | bands scored: %d of %d"
          % (len(cells), verdict["n_checked"], verdict["n_bands"]))
    print()
    for name in BANDS:
        v = verdict["bands"].get(name)
        if v is None:
            print("  %-12s NOT SCORED (no cell produced it)" % name)
            continue
        print("  %-12s %s" % (name, "PASS" if v["passed"] else "FAIL"))
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
    print("ALL SIX BANDS PASS" if (verdict["passed"] and complete)
          else "NOT A PASS: %s" % ("a band failed" if complete
                                   else "not every band was scored"))
    return 0 if (verdict["passed"] and complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
