#!/usr/bin/env python3
"""Check every number in main.tex against a row of the generated tables.

The preprint's figures are supposed to come from the canonical tables and
from REVIEW_FACTS.md, both of which are generated from the committed
result JSONs and have verifiers of their own. "Supposed to" is the part
this script removes: it extracts every number in ``main.tex``, looks for
it in those two documents, and reports what it cannot find.

Three findings, kept separate because they mean different things:

  NO SOURCE   the number appears in no row of either document. Either it
              is not a measurement (a batch size, a seed, a year) and
              belongs to the skip list below, or it is a measurement that
              nothing generated -- which is the finding.
  MISMATCH    a number in the documents is CLOSE but outside the
              tolerance the preprint's own printed precision allows. The
              likely intended source is named with the difference.
  ROUNDING    the number matches, but its printed precision is coarser
              than a difference the sentence around it asserts, or
              rounding moves it across zero. The claim and the
              measurement then disagree even though the number does not.

The tolerance is not global. Each number gets **half a unit in its own
last printed place**: ``3.11`` admits 0.005, ``81.554`` admits 0.0005,
``45`` admits 0.5. That is the most a correctly rounded figure can differ
from its source, so anything beyond it is a different number and not a
rounding. The report states the tolerance per number.

What is deliberately NOT checked: prose. This script reads numbers. A
sentence can be wrong with every digit right, and no extractor finds
that.

    python docs/preprint/verify_main_tex.py --tables <dir-with-CANONICAL_TABLES.md>

``--tables`` is required rather than defaulted, so this file names no
location outside its own repository. The canonical tables are deposited
with the preprint in its public data archive; point this at wherever
they are.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.tex"
FACTS = HERE / "REVIEW_FACTS.md"

#: A number is a candidate measurement unless it is one of these. Every
#: entry is a pattern over the ORIGINAL text around the number, and every
#: one of them is here because it fired on a real line of main.tex.
SKIP_CONTEXT = (
    # Citation keys, labels and references carry digits that are names.
    (re.compile(r"\\cite[tp]?\{[^}]*$"), "inside a citation key"),
    (re.compile(r"\\(?:label|ref|eqref|autoref)\{[^}]*$"), "inside a label"),
    (re.compile(r"\\url\{[^}]*$"), "inside a URL"),
    (re.compile(r"\\texttt\{[^}]*$"), "inside a literal"),
    (re.compile(r"\\begin\{tabular\}\{[^}]*$"), "a column spec"),
    (re.compile(r"\\(?:tabcolsep|arraystretch|baselineskip|textwidth|"
                r"columnwidth|hspace|vspace|setlength|includegraphics)"
                r"[^\n]*$"), "a layout length"),
)

#: Numbers that are not measurements wherever they appear: model sizes
#: and names, dataset sizes, versions, seeds, task counts. Each is a FACT
#: about the setup, stated in the protocol and not derived from a result
#: row, so looking for it in a results table would report a false gap.
#: The value is why it is exempt -- an exemption with no reason is how a
#: real number ends up hidden in a skip list.
SETUP_NUMBERS = {
    # Numbers that carry a unit and still are not results: they are the
    # protocol, stated once. An exemption with no reason is how a real
    # number ends up hidden in a skip list, so each gives one.
    "50": "a pruning ratio, stated as the recipe, not a measured value",
    "33": "a pruning ratio, stated as the recipe",
    "25": "a pruning ratio, stated as the recipe",
    "40": "a pruning ratio, stated as the recipe",
    "2048": "the OPT sequence length, in tokens",
    "50000": "the ImageNet-1k validation split size, in images",
    "50{,}000": "the ImageNet-1k validation split size, in images",
    "500": "the FI probe size, in images, stated as the protocol",
    "64": "the FI probe's small size, in images",
    "1030": "the edge count of a 144-head graph at the fixed density",
    "0.25": "the shipped CRITICAL threshold",
    "0.10": "the FI edge density",
    "32M": "the OPT recovery budget, in tokens",
}


#: A number, with the unit that follows it if there is one. The first
#: draft forbade a trailing word character, which made every
#: ``2.044x`` and ``11.79 ppl`` in the generated documents invisible and
#: turned each of them into a mismatch against an unrelated row.
#: No trailing lookahead at all. The first draft forbade a trailing
#: word character, which made every ``2.044x`` and ``11.79 ppl`` in
#: the generated documents invisible and turned each of them into a
#: mismatch against an unrelated row. group(1) is the number; the
#: caller reads whatever follows it.
NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)")
#: What makes a number a candidate MEASUREMENT rather than a setting.
#: Either a unit follows it, or it sits in a numeric column of a table
#: row. Everything else is reported as unclassified, with a count, so
#: the exclusion is visible instead of being a list someone maintains.
#: A number inside an identifier: the tail of a grant code, a model
#: name, a revision. Recognised by the hyphen that joins it to a
#: preceding alphanumeric, not by its value.
IDENTIFIER_TAIL = re.compile(r"[A-Za-z0-9]-$")
#: A version number. The names are the software this campaign pins, and
#: a version is a fact about the environment rather than a measurement;
#: REVIEW_FACTS records the freeze.
VERSION_BEFORE = re.compile(
    r"(?:oneMKL|ONNX Runtime|Runtime|PyTorch|ROCm|transformers|tokenizers|"
    r"lm-evaluation-harness|lm-eval|datasets|accelerate|Python|CUDA|"
    r"bitsandbytes|numpy)\s*v?$", re.I)
#: A line that is ABOUT a measured quantity. Permissive deliberately: a
#: version number on such a line is then reported as having no source,
#: where it can be seen, rather than skipped where it cannot.
MEASURED_LINE = re.compile(
    r"perplexit|accurac|points?\b|\bpp\b|\bppl\b|cost|spread|retain|"
    r"speedup|GCD|GPU-h|probe|drop|margin|interval|seeds?\b|"
    r"throughput|latency|edges?\b|FI\b", re.I)
MEASURED_UNIT = re.compile(
    r"^\s*(?:\\[,%]|~|\$)?\s*"
    r"(?:%|\\%|pp\b|ppl\b|points?\b|perplexity|x\b|\\times|GiB|GB|"
    r"GCD-h|GPU-h|GPU-hours?|ms\b|samples?/s)")


def places(text: str) -> int:
    """Decimal places in the number AS PRINTED."""
    t = text.replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


#: A correctly rounded number differs from its source by at most half a
#: unit in the last place -- EXACTLY that, half the time. In binary
#: floating point the subtraction then lands a few 1e-15 above the
#: bound, so the comparison carries an epsilon. Without it, 82.48
#: against a source 82.475 was reported as a mismatch.
_EPS = 1e-9


def tol_of(text: str) -> float:
    """Half a unit in the last printed place, plus a float epsilon."""
    return 0.5 * 10 ** (-places(text)) + _EPS


def to_float(text: str) -> float | None:
    t = text.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def strip_tex(line: str) -> str:
    """Remove the TeX that is never a measurement, keeping offsets rough.

    ``{,}`` first: it is how TeX writes a thousands separator, and
    leaving it in made ``2{,}717.7`` extract as 717.7 and ``50{,}000``
    as 50 and 000. Both were then reported as having no source.
    """
    s = line.replace("{,}", "").replace("{,}", "")
    s = re.sub(r"\\cite[tp]?\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:label|ref|eqref|autoref)\{[^}]*\}", " ", s)
    s = re.sub(r"\\url\{[^}]*\}", " ", s)
    return s


def extract_main(path: Path) -> list:
    """Every candidate measurement in main.tex, with its context."""
    out = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if line.lstrip().startswith("%"):
            continue
        cleaned = strip_tex(line)
        for m in NUM.finditer(cleaned):
            text = m.group(1)
            before = cleaned[:m.start()]
            skipped = None
            for rx, why in SKIP_CONTEXT:
                if rx.search(before):
                    skipped = why
                    break
            val = to_float(text)
            if val is None:
                continue
            after = cleaned[m.end(1):m.end(1) + 14]
            if IDENTIFIER_TAIL.search(before[-2:]) or \
                    VERSION_BEFORE.search(before[-40:]):
                skipped = ("part of an identifier or a version number"
                           if IDENTIFIER_TAIL.search(before[-2:])
                           else "a pinned software version")
            # A table row's numeric columns are measurements even without
            # a unit: the caption carries the unit for the whole column.
            in_table_col = ("&" in cleaned
                            and re.fullmatch(r"[\s$+\-()\\a-z.0-9,]*",
                                             after.split("&")[0] or ""))
            out.append({"line": i, "text": text, "value": val,
                        "places": places(text), "tol": tol_of(text),
                        "context": line.strip()[:400],
                        "near": cleaned[max(0, m.start() - 60):
                                        m.end() + 60].strip(),
                        "measured": bool(MEASURED_UNIT.match(after))
                                    or bool(in_table_col)
                                    or bool(MEASURED_LINE.search(cleaned)),
                        "skipped": skipped})
    return out


def extract_sources(paths: list) -> list:
    """Every number in the generated documents, with where it came from."""
    rows = []
    for p in paths:
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(),
                                 1):
            for m in NUM.finditer(line):
                v = to_float(m.group(1))
                if v is None:
                    continue
                rows.append({"doc": p.name, "line": i, "value": v,
                             "text": m.group(1),
                             "row": line.strip()[:300]})
    return rows


def _assertion_precision(near: str) -> float | None:
    """The smallest difference the sentence around a number asserts.

    A sentence that says two figures differ, or that one is above
    another, is making a claim whose resolution is bounded by the
    coarsest number in it. If the printed precision cannot express the
    difference, the number is right and the claim is not checkable from
    it. Only a few shapes are recognised; the rest are not reported, so
    this finding under-reports rather than invents.
    """
    pair = re.findall(r"(\d+(?:\.\d+)?)\s*(?:against|to|and|vs\.?|versus)\s*"
                      r"(\d+(?:\.\d+)?)", near)
    for a, b in pair:
        fa, fb = to_float(a), to_float(b)
        if fa is None or fb is None or fa == fb:
            continue
        d = abs(fa - fb)
        coarse = max(tol_of(a), tol_of(b))
        if d < 2 * coarse:
            return d
    return None


def verify(main_path: Path, source_paths: list, strict_setup: bool) -> dict:
    nums = extract_main(main_path)
    srcs = extract_sources(source_paths)
    if not srcs:
        raise SystemExit(
            "no source numbers found. Point --tables at the directory that "
            "holds CANONICAL_TABLES.md.")

    findings = {"ok": [], "no_source": [], "mismatch": [], "rounding": [],
                "skipped": [], "setup": [], "unclassified": []}
    for n in nums:
        if n["skipped"]:
            findings["skipped"].append(n)
            continue
        key = n["text"].replace(",", "")
        if key in SETUP_NUMBERS and not strict_setup:
            n["why"] = SETUP_NUMBERS[key]
            findings["setup"].append(n)
            continue
        if not n["measured"] and not strict_setup:
            findings["unclassified"].append(n)
            continue

        tol = n["tol"]
        hits = [s for s in srcs if abs(s["value"] - n["value"]) <= tol]
        # The documents carry costs as signed deltas and the preprint
        # sometimes states them as magnitudes ("costs 0.21 points" for a
        # row that reads -0.21pp). An absolute-value match is a match,
        # and is reported as one so the sign convention stays visible.
        signed = False
        if not hits:
            hits = [s for s in srcs
                    if abs(abs(s["value"]) - abs(n["value"])) <= tol]
            signed = bool(hits)
        if hits:
            best = min(hits, key=lambda s: abs(abs(s["value"])
                                               - abs(n["value"])))
            n["source"] = best
            n["abs_match_only"] = signed
            d = _assertion_precision(n["near"])
            if d is not None and d < 2 * tol:
                n["assertion_delta"] = d
                findings["rounding"].append(n)
            else:
                findings["ok"].append(n)
            continue

        # Nothing within tolerance. Is something close enough that it is
        # plainly the intended row with a different value?
        near = min(srcs, key=lambda s: abs(abs(s["value"])
                                           - abs(n["value"])))
        gap = abs(abs(near["value"]) - abs(n["value"]))
        if gap <= 10 * tol and gap > 0:
            n["nearest"] = near
            n["gap"] = gap
            findings["mismatch"].append(n)
        else:
            n["nearest"] = near
            n["gap"] = gap
            findings["no_source"].append(n)
    return findings


def render(f: dict, main_path: Path) -> str:
    L = ["=== verify_main_tex: %s ===" % main_path.name, ""]
    n_checked = len(f["ok"]) + len(f["mismatch"]) + len(f["rounding"]) \
        + len(f["no_source"])
    L.append("  %d number(s) checked against the generated documents"
             % n_checked)
    L.append("  %d matched, %d with no source, %d mismatched, "
             "%d rounding question(s)"
             % (len(f["ok"]), len(f["no_source"]), len(f["mismatch"]),
                len(f["rounding"])))
    L.append("  %d skipped as TeX, %d as protocol constants, "
             "%d unclassified (no unit and not a numeric table column)"
             % (len(f["skipped"]), len(f["setup"]),
                len(f["unclassified"])))

    if f["mismatch"]:
        L += ["", "--- MISMATCH: the nearest generated number differs by "
                  "more than the preprint's own printed precision allows",
              ""]
        for n in sorted(f["mismatch"], key=lambda x: -x["gap"]):
            L.append("  main.tex:%d  %s  (tolerance +-%g)"
                     % (n["line"], n["text"], n["tol"]))
            L.append("      nearest: %s in %s:%d, differs by %.6g"
                     % (n["nearest"]["text"], n["nearest"]["doc"],
                        n["nearest"]["line"], n["gap"]))
            L.append("      row: %s" % n["nearest"]["row"])
            L.append("      near: ...%s..." % n["near"])
            L.append("")

    if f["no_source"]:
        L += ["", "--- NO SOURCE: no row of either generated document "
                  "carries this number", ""]
        for n in sorted(f["no_source"], key=lambda x: x["line"]):
            L.append("  main.tex:%d  %s  (tolerance +-%g)"
                     % (n["line"], n["text"], n["tol"]))
            L.append("      near: ...%s..." % n["near"])
            L.append("")

    if f["rounding"]:
        L += ["", "--- ROUNDING: the printed precision is coarser than a "
                  "difference the sentence asserts", ""]
        for n in sorted(f["rounding"], key=lambda x: x["line"]):
            L.append("  main.tex:%d  %s  (tolerance +-%g, the sentence "
                     "asserts a difference of %g)"
                     % (n["line"], n["text"], n["tol"],
                        n["assertion_delta"]))
            L.append("      near: ...%s..." % n["near"])
            L.append("")

    abs_only = [n for n in f["ok"] if n.get("abs_match_only")]
    if abs_only:
        L += ["", "--- SIGN: matched on magnitude only. The documents "
                  "carry a signed cost where the preprint states a "
                  "magnitude; both can be right, and this lists them so "
                  "the convention stays visible.", ""]
        for n in sorted(abs_only, key=lambda x: x["line"])[:40]:
            L.append("  main.tex:%d  %s  <- %s in %s:%d"
                     % (n["line"], n["text"], n["source"]["text"],
                        n["source"]["doc"], n["source"]["line"]))
        if len(abs_only) > 40:
            L.append("  ... and %d more" % (len(abs_only) - 40))
        L.append("")

    if f["unclassified"]:
        L += ["", "--- UNCLASSIFIED: no unit follows these and they are not "
                  "in a numeric table column, so they are counts, ratios, "
                  "versions or sizes rather than measurements. Listed, not "
                  "excused: --strict-setup checks them too.", ""]
        seen = {}
        for n in f["unclassified"]:
            seen.setdefault(n["text"], []).append(n["line"])
        for t in sorted(seen, key=lambda x: -len(seen[x]))[:30]:
            L.append("  %-10s x%-3d  lines %s"
                     % (t, len(seen[t]),
                        ",".join(str(x) for x in sorted(seen[t])[:8])
                        + (" ..." if len(seen[t]) > 8 else "")))
        if len(seen) > 30:
            L.append("  ... and %d more distinct value(s)" % (len(seen) - 30))
        L.append("")

    L += ["",
          "  A number listed under NO SOURCE is not necessarily wrong: it "
          "may be a setup constant this script does not know. The "
          "exemption list in SETUP_NUMBERS states a reason per entry, and "
          "--strict-setup checks those too.",
          ""]
    bad = len(f["no_source"]) + len(f["mismatch"])
    L.append("  %s" % ("EVERY NUMBER HAS A SOURCE" if not bad
                       else "%d number(s) need attention" % bad))
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main", default=str(MAIN))
    p.add_argument("--tables", default=os.environ.get("SAL_CANONICAL_TABLES"),
                   help="directory holding CANONICAL_TABLES.md; or set "
                        "SAL_CANONICAL_TABLES")
    p.add_argument("--facts", default=str(FACTS))
    p.add_argument("--strict-setup", action="store_true",
                   help="also require a source for the setup constants")
    a = p.parse_args(argv)
    if not a.tables:
        p.error("--tables is required (or set SAL_CANONICAL_TABLES)")

    srcs = [Path(a.tables) / "CANONICAL_TABLES.md", Path(a.facts)]
    missing = [str(s) for s in srcs if not s.is_file()]
    if missing:
        p.error("not found: %s" % ", ".join(missing))

    f = verify(Path(a.main), srcs, a.strict_setup)
    print(render(f, Path(a.main)))
    return 1 if (f["no_source"] or f["mismatch"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
