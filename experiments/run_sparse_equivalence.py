#!/usr/bin/env python3
"""Run the SparseGPT / Wanda equivalence test and archive what it proves.

``tests/test_exp33_sparse_equivalence.py`` has two tiers. The invariants run
everywhere; the four tests that compare our pruners against the **upstream
reference code** are skipped unless ``SAL_SPARSE_REFS`` points at clones of
``IST-DASLab/sparsegpt`` and ``locuslab/wanda``. Until this script existed
those four had never run: not in CI, which has no clones, and not in any
LUMI job. The preprint nonetheless stated the comparison as done, which is
the gap this closes.

A pass is only meaningful with the upstream commit it was taken against, so
the archive records the remote, the commit hash and the commit date of each
reference, the interpreter and torch versions, the tolerances asserted, and
the full pytest output. Re-running against a newer upstream commit is then
a different measurement, and the JSON says which one was made.

    git clone --depth 1 https://github.com/IST-DASLab/sparsegpt $REFS/sparsegpt
    git clone --depth 1 https://github.com/locuslab/wanda       $REFS/wanda
    SAL_SPARSE_REFS=$REFS python experiments/run_sparse_equivalence.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEST = "tests/test_exp33_sparse_equivalence.py"
REMOTES = {
    "sparsegpt": "https://github.com/IST-DASLab/sparsegpt",
    "wanda": "https://github.com/locuslab/wanda",
}

#: What the tier-2 tests assert, so the archive states the claim it backs
#: rather than only "passed".
TOLERANCES = {
    "wanda.scaler_row": "allclose, atol=1e-6 (activation-norm accumulation)",
    "wanda.mask": "torch.equal on the zero pattern",
    "wanda.weights": "allclose, atol=1e-6",
    "sparsegpt.H": "allclose, atol=1e-5 (Hessian)",
    "sparsegpt.mask": "torch.equal on the zero pattern",
    "sparsegpt.weights": "allclose, atol=1e-5",
    "nm_2of4": "torch.equal on the zero pattern, both pruners",
}


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return "unavailable (%s)" % exc


def reference(refs: Path, name: str) -> dict:
    d = refs / name
    if not d.is_dir():
        return {"name": name, "present": False,
                "expected_at": str(d), "remote": REMOTES[name]}
    return {
        "name": name,
        "present": True,
        "remote": git(d, "config", "--get", "remote.origin.url")
                  or REMOTES[name],
        "commit": git(d, "rev-parse", "HEAD"),
        "commit_date": git(d, "log", "-1", "--format=%cI"),
        "subject": git(d, "log", "-1", "--format=%s"),
        "describe": git(d, "describe", "--always", "--tags", "--dirty"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", default=os.environ.get("SAL_SPARSE_REFS"),
                   help="directory holding the sparsegpt/ and wanda/ clones")
    p.add_argument("--output", default="results/lumi_v060/exp33_equivalence")
    args = p.parse_args(argv)

    if not args.refs:
        print("SAL_SPARSE_REFS is not set and --refs was not given. The "
              "tier-2 tests would skip, and an archive of four skipped "
              "tests proves nothing, so this script refuses to write one.")
        return 2
    refs = Path(args.refs)
    rec = {name: reference(refs, name) for name in REMOTES}
    missing = [n for n, r in rec.items() if not r["present"]]
    if missing:
        print("missing reference clone(s): %s" % ", ".join(missing))
        for n in missing:
            print("  git clone --depth 1 %s %s" % (REMOTES[n], refs / n))
        return 2

    env = dict(os.environ, SAL_SPARSE_REFS=str(refs), PYTHONUTF8="1")
    cmd = [sys.executable, "-m", "pytest", TEST, "-v", "--tb=short",
           "-p", "no:warnings"]
    # What goes in the archive is the reproducible form. sys.executable is
    # an absolute path that carries the account name on some machines, and
    # this repository is public.
    printable = " ".join(["python", "-m", "pytest", TEST, "-v", "--tb=short",
                          "-p", "no:warnings"])
    started = datetime.now(timezone.utc)
    run = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True,
                         text=True)
    finished = datetime.now(timezone.utc)
    out = run.stdout + ("\n" + run.stderr if run.stderr.strip() else "")

    # Per-test outcomes, parsed from the -v lines rather than counted.
    outcomes = {}
    for line in out.splitlines():
        if "::" in line and any(k in line for k in
                                ("PASSED", "FAILED", "SKIPPED", "ERROR")):
            name = line.split("::", 1)[1].split()[0]
            for k in ("PASSED", "FAILED", "SKIPPED", "ERROR"):
                if k in line:
                    outcomes[name] = k
                    break
    tier2 = {k: v for k, v in outcomes.items() if "matches_the_reference" in k}

    doc = {
        "experiment": "exp33_equivalence",
        "what": ("our vendored SparseGPT and Wanda against the upstream "
                 "reference implementations, layer by layer on one linear "
                 "layer and a fixed calibration set"),
        "why": ("the preprint states this comparison as done; before this "
                "run the four tier-2 tests had never executed, because they "
                "skip unless SAL_SPARSE_REFS is set and neither CI nor any "
                "LUMI job sets it"),
        "test_file": TEST,
        "command": printable,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "exit_code": run.returncode,
        "passed": run.returncode == 0,
        "references": rec,
        "tolerances_asserted": TOLERANCES,
        "outcomes": outcomes,
        "tier2_outcomes": tier2,
        "n_tests": len(outcomes),
        "n_tier2": len(tier2),
        "tier2_all_passed": bool(tier2) and set(tier2.values()) == {"PASSED"},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _torch_version(),
            "pytest": _pytest_version(),
        },
        "caveat": ("a layer-level equivalence, on CPU, at one sparsity and "
                   "one calibration set. It shows the pruning arithmetic "
                   "matches upstream; it does not reproduce the papers' "
                   "end-to-end perplexities, which are measured on models "
                   "and a pipeline this campaign does not run."),
    }

    outdir = REPO / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exp33_equivalence.json").write_text(
        json.dumps(doc, indent=1), encoding="utf-8")
    (outdir / "pytest.log").write_text(_scrub(out), encoding="utf-8")

    print("=== sparse equivalence ===")
    for name, r in rec.items():
        print("  %-10s %s @ %s (%s)"
              % (name, r["remote"], r["commit"][:12], r["commit_date"][:10]))
    print("  %d test(s), %d against the references: %s"
          % (len(outcomes), len(tier2),
             "all passed" if doc["tier2_all_passed"] else str(tier2)))
    print("  exit %d -> %s" % (run.returncode,
                               "PASS" if doc["passed"] else "FAIL"))
    print("\nwrote %s" % (outdir / "exp33_equivalence.json"))
    if not doc["passed"]:
        print(out[-2000:])
    return 0 if doc["passed"] else 1


def _scrub(text: str) -> str:
    """Replace machine-specific absolute paths with placeholders.

    pytest's header prints the interpreter path and the rootdir. Neither
    tells a reader anything about the measurement, and on this machine the
    first carries the account name.
    """
    text = text.replace(str(REPO), "<repo>")
    text = text.replace(str(REPO).replace("\\", "/"), "<repo>")
    text = text.replace(sys.executable, "<python>")
    text = text.replace(sys.executable.replace("\\", "/"), "<python>")
    # Anything else that still looks like a Windows or POSIX home directory.
    text = re.sub(r"[A-Za-z]:\\\\Users\\\\[^\\\\\\s]+", r"<home>", text)
    text = re.sub(r"/home/[^/\s]+", "<home>", text)
    return text


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "unavailable"


def _pytest_version() -> str:
    try:
        import pytest
        return pytest.__version__
    except ImportError:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
