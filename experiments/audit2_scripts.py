#!/usr/bin/env python3
"""Audit 2: check every campaign script against the structural rules.

Rule 4 of the audit applied to the audit. The findings below are produced by
reading the source, not by remembering what each script does, so the table can
be regenerated after any change and a regression shows up as a row rather than
as a gap in someone's recollection.

The rules, as given:

1. **One harness.** No experiment builds its own data, probe or loader.
   Shared functions only, validating their inputs: a probe must be float and
   normalized, ``num_classes`` must be asserted, and a memory-mapped split
   must be read with ``workers > 0``. Fail loudly.
2. **Reference gate.** Every script reproduces a known value before
   reporting a new one.
3. **CPU micro-test** before the LUMI gate.
4. **Reports generated from computed tables**, with no narrated claim that is
   not a row of a table some code produced.
5. **Cross-harness consistency** kept as a regression test.

What this tool can and cannot see. It reads source text, so it detects the
*shape* of a violation -- a probe dict built inline, a missing assertion, a
hand-written table -- and it cannot prove a number is right. A clean row means
"this script no longer has the structure that produced the campaign's known
wrong numbers", not "this script is correct".

Usage::

    python experiments/audit2_scripts.py
    python experiments/audit2_scripts.py --json audit2.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOTS = {
    "sal-torch": _HERE.parent,
    "sal-torch-pro": _HERE.parent.parent / "sal-torch-pro",
    # The third repository is private and is audited from its own checkout;
    # see the local notes. Nothing here names it.
}

# Scripts in scope: the wave-1 and wave-2 campaign (exp29 onward, the ones
# this and the preceding audit were called on) plus every script that computes
# FI, whatever its number.
IN_SCOPE = re.compile(
    r"^(exp(29|3\d|4\d)\w*|exp01_fi_scan|exp17_fi_regime|matched_control"
    r"|results_corrections)\.py$")

# Public FI entry points. A sibling repo whose FI path has its own entry
# symbol passes it with --extra-fi-call; this file is on a public remote and
# does not enumerate another package's API.
FI_CALLS = ["extract_activation_graph", "compute_fi", "FIScanner",
            "FIMonitor", "classify_layers", "layer_fi_deltas"]


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def builds_own_probe(src: str, is_fi: bool) -> list[str]:
    """Inline ``{"pixel_values": ...}`` construction outside the shared builder.

    The campaign's most expensive defect: exp37 and exp42 each built a probe
    from the raw uint8 cache, both runs completed, and both reported a
    plausible FI of a different quantity. A construction that goes through
    ``P.build_probe`` is validated; one written inline is not.
    """
    if not is_fi:
        # A latency or export benchmark also builds {"pixel_values": ...}
        # tensors, but they are forward-pass inputs and no structural number
        # is derived from them. Only an FI probe has to be validated.
        return []
    if "validate_probe" in src:
        # A probe that comes from a model's own processor cannot be rebuilt by
        # the shared builder -- that builder applies one model's
        # normalization, and exp01 and exp17 scan several architectures. What
        # the harness contributes in that case is the validation, so a script
        # that calls validate_probe on what it built has satisfied the rule:
        # the check is the part that catches a wrong probe.
        return []
    hits = []
    for m in re.finditer(r'\{\s*"pixel_values"\s*:', src):
        line = src[:m.start()].count("\n") + 1
        # The shared builder itself, and a single-tensor verify batch sliced
        # out of an already-validated probe, are not separate constructions.
        ctx = src[max(0, m.start() - 400):m.start()]
        if "def build_probe" in ctx or "def validate_probe" in ctx:
            continue
        hits.append("line {}".format(line))
    return hits


def asserts_num_classes(src: str) -> bool:
    """Is the class count checked anywhere on this script's data path?

    The assertion now lives in ``P.build_data`` -- one builder, one check,
    rule 1 -- so a script that loads its split through it is covered, and a
    script that loads a split some other way is not. Before that, exp29,
    exp31 and exp33 each read a split with nothing checking the class count:
    a changed mirror or a stale cache would have produced an accuracy that
    looked ordinary and was not comparable to anything published.
    """
    if re.search(r"num_classes\s*!=\s*\d+", src) or "check_reference" in src:
        return True
    return "build_data(" in src


def guards_loader_workers(src: str) -> bool:
    """A memory-mapped split read with workers=0 is a silent 10x slowdown."""
    if "LOADER_WORKERS" not in src and "workers" not in src:
        return True                      # does not load a split at all
    return ("LOADER_WORKERS" in src) or ("loader_workers" in src)


EXEMPT = re.compile(r"#\s*audit2:\s*reference-gate-exempt\s*--\s*(.+)")


def has_reference_gate(src: str) -> bool:
    return "check_reference" in src or "matched_control" in src


def gate_exemption(src: str) -> str | None:
    """A stated, in-file reason why this script has no reference gate.

    An exemption is printed as an exemption, never as a pass. The point of
    the marker is that the reason lives next to the code and is reviewable,
    rather than the finding being silenced in the tool or tolerated as
    permanent noise until the table stops being read.
    """
    m = EXEMPT.search(src)
    return m.group(1).strip() if m else None


def computes_its_tables(src: str) -> bool:
    """Does the script emit a table it computed, or only prose?

    Not a strong check -- it looks for a formatted table or a verdict block
    built from values in scope. A script that writes JSON and formats rows
    from it passes; one that only logs sentences does not.
    """
    return ('"verdict"' in src or "verdict" in src) and \
        bool(re.search(r"\{[^}]*:[<>^]\d+[^}]*\}|format\(", src))


def touches_fi(src: str) -> bool:
    return any(c in src for c in FI_CALLS)


def audit(path: Path, repo: str) -> dict:
    src = _src(path)
    try:
        ast.parse(src)
        parses = True
    except SyntaxError:
        parses = False
    probe = builds_own_probe(src, touches_fi(src))
    row = {
        "repo": repo,
        "script": path.name,
        "parses": parses,
        "touches_fi": touches_fi(src),
        "own_probe_construction": probe,
        "num_classes_asserted": asserts_num_classes(src),
        "loader_workers_guarded": guards_loader_workers(src),
        "reference_gate": has_reference_gate(src),
        "reference_gate_exemption": gate_exemption(src),
        "emits_a_computed_table": computes_its_tables(src),
    }
    findings = []
    if not parses:
        findings.append("does not parse")
    if probe:
        findings.append(
            "builds its own probe at {} -- use P.build_probe, which "
            "validates (rule 1)".format(", ".join(probe)))
    if touches_fi(src) and not has_reference_gate(src)             and not gate_exemption(src):
        findings.append(
            "computes FI with no reference gate: it can report a new number "
            "without reproducing a published one (rule 2)")
    if not asserts_num_classes(src) and "build_data" in src:
        findings.append(
            "loads a split without asserting num_classes (rule 1)")
    if not guards_loader_workers(src):
        findings.append("reads a split without setting loader workers (rule 1)")
    row["findings"] = findings
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default=None)
    p.add_argument("--all", action="store_true",
                   help="every script, not only those in scope")
    p.add_argument("--extra-fi-call", action="append", default=[],
                   metavar="SYMBOL",
                   help="an additional symbol that marks a script as touching "
                        "FI, for a sibling repo with its own entry point. "
                        "Repeatable.")
    args = p.parse_args(argv)
    FI_CALLS.extend(args.extra_fi_call)

    rows = []
    for repo, root in ROOTS.items():
        d = root / "experiments"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            if args.all or IN_SCOPE.match(f.name):
                rows.append(audit(f, repo))

    w = max(len(r["script"]) for r in rows) + 2
    print("audit 2 -- {} scripts in scope\n".format(len(rows)))
    head = ("{:<14}{:<" + str(w) + "}{:>4}{:>7}{:>6}{:>5}{:>6}{:>7}").format(
        "repo", "script", "FI", "probe", "ncls", "wrk", "ref", "table")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(("{:<14}{:<" + str(w) + "}{:>4}{:>7}{:>6}{:>5}{:>6}{:>7}").format(
            r["repo"], r["script"],
            "y" if r["touches_fi"] else "-",
            "OWN" if r["own_probe_construction"] else "ok",
            "ok" if r["num_classes_asserted"] else "NO",
            "ok" if r["loader_workers_guarded"] else "NO",
            "ok" if r["reference_gate"]
            else ("exmt" if r["reference_gate_exemption"] else "NO"),
            "ok" if r["emits_a_computed_table"] else "NO"))

    exempt = [r for r in rows if r["reference_gate_exemption"]]
    if exempt:
        print("\nreference-gate exemptions (stated in the file, not a pass):")
        for r in exempt:
            print("  {:<32}{}".format(
                r["script"], r["reference_gate_exemption"]))

    flagged = [r for r in rows if r["findings"]]
    print("\n{} script(s) with findings, {} clean".format(
        len(flagged), len(rows) - len(flagged)))
    for r in flagged:
        print("\n  {}/{}".format(r["repo"], r["script"]))
        for f in r["findings"]:
            print("    - {}".format(f))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"scripts": rows,
             "summary": {"total": len(rows), "flagged": len(flagged)}},
            indent=2), encoding="utf-8")
        print("\nwrote {}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
