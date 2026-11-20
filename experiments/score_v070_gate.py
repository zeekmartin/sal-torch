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
        for band in BANDS_BY_CELL.get(key, ()):
            out[band] = {"per_seed": v["values"], "cost": mean_cost,
                         "n_cells": len(v["cells"])}
    return out


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
    provs = [c.get("provenance") for c in cells if c.get("provenance")]
    prov = provs[0] if provs else None

    # Every cell must agree on the code it used. A batch scored across two
    # different packages is not one measurement.
    hashes = {(c.get("provenance") or {}).get("computed", {})
              .get("aggregate_sha256") for c in cells}
    hashes.discard(None)
    mixed = len(hashes) > 1

    verdict = check_all(scored, provenance=prov, target=args.target,
                        repo=HERE.parent)
    if mixed:
        verdict["passed"] = False
        for v in verdict["bands"].values():
            v["passed"] = False
            v["why"] = list(v["why"]) + [
                "provenance: the cells were produced by more than one package "
                "(%d distinct hashes), so they are not one measurement"
                % len(hashes)]

    if args.json:
        print(json.dumps({"verdict": verdict, "scored": scored,
                          "n_cells": len(cells), "mixed_packages": mixed},
                         indent=1, default=str))
        return 0 if verdict["passed"] else 1

    print("=== v0.7.0 gate, scored against the pre-registration ===")
    print("cells read: %d | bands scored: %d of %d"
          % (len(cells), verdict["n_checked"], verdict["n_bands"]))
    pv = verdict.get("provenance")
    if pv:
        print("code under test: %s" % pv["why"])
    else:
        print("code under test: NOTHING RECORDED, so no band can pass")
    if mixed:
        print("MIXED PACKAGES: %d distinct package hashes across the cells"
              % len(hashes))
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
