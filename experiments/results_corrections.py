#!/usr/bin/env python3
"""Read a committed result JSON with its corrections applied.

Measured data is never mutated. A result file on disk stays byte-for-byte what
the job wrote; ``results/lumi_v060/CORRECTIONS.json`` records which *derived*
fields in it are wrong and what replaces them, and :func:`load_result` applies
them in memory. A table built from JSON then agrees with the write-ups.

Why this exists. ``exp32_ctrl_s{42,123,456}/exp31_dense_control.json`` carries
``vs_this_control_pp`` values of -2.269 / -2.831 / -4.411: exp29's **five**-epoch
ratio arms scored against a **two**-epoch dense control. The matched costs are
-1.30 / -1.93 / -3.91, so the field is wrong by about a point on three headline
cells. ``experiments/exp31_dense_control.py`` now refuses that restatement, but
the fix landed after the jobs ran, so the bad values are live in committed data
with nothing in the file to flag them.

Rewriting them in place would destroy the audit trail and make the repository
disagree with itself about what was measured. So the data stays and the
correction travels beside it.

Usage::

    from results_corrections import load_result, audit

    d = load_result("results/lumi_v060/exp32_ctrl_s42/exp31_dense_control.json")
    d["exp29_restated"]["epochs_matched"]           # False
    "vs_this_control_pp" in d["exp29_restated"]["ratios"]["0.50"]   # False
    d["_corrections_applied"]                       # the ids, in order

    audit()        # every correction, and whether its target still matches

Command line::

    python experiments/results_corrections.py --audit
    python experiments/results_corrections.py results/lumi_v060/exp31_s42/exp31_dense_control.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable

__all__ = ["load_result", "load_corrections", "audit", "CorrectionError",
           "REPO_ROOT", "CORRECTIONS_PATH"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CORRECTIONS_PATH = REPO_ROOT / "results" / "lumi_v060" / "CORRECTIONS.json"

_VALID_OPS = {"set", "remove", "annotate"}
# A derived-value expression in a correction: "=(a/b - c/d)*100". Only the two
# forms actually needed are supported, and anything else raises, because a
# general expression evaluator in a data file is how a sidecar becomes code.
_EXPR = re.compile(r"^=\((?P<l>[\w/]+) - (?P<r>[\w/]+)\)\*100$")


class CorrectionError(RuntimeError):
    """A correction does not apply to the file it names."""


# --------------------------------------------------------------------------- #
# pointers
# --------------------------------------------------------------------------- #

def _split(pointer: str) -> list[str]:
    parts = [p for p in pointer.split("/") if p != ""]
    if not parts:
        raise CorrectionError(f"empty pointer: {pointer!r}")
    return parts


def _resolve(doc: Any, parts: Iterable[str]) -> list[tuple[Any, str]]:
    """Every (container, key) a pointer selects. ``*`` fans out over a dict."""
    here: list[Any] = [doc]
    parts = list(parts)
    for depth, key in enumerate(parts[:-1]):
        nxt: list[Any] = []
        for node in here:
            if key == "*":
                if not isinstance(node, dict):
                    continue
                nxt.extend(node.values())
            elif isinstance(node, dict) and key in node:
                nxt.append(node[key])
        here = nxt
        if not here:
            raise CorrectionError(
                f"pointer {'/'.join(parts)} does not resolve: no {key!r} at depth {depth}")
    leaf = parts[-1]
    out: list[tuple[Any, str]] = []
    for node in here:
        if leaf == "*":
            if isinstance(node, dict):
                out.extend((node, k) for k in list(node))
        elif isinstance(node, dict):
            out.append((node, leaf))
    if not out:
        raise CorrectionError(f"pointer {'/'.join(parts)} selected nothing")
    return out


def _lookup(doc: Any, path: str) -> Any:
    node = doc
    for key in _split(path):
        if not isinstance(node, dict) or key not in node:
            raise CorrectionError(f"cannot read {path!r}: missing {key!r}")
        node = node[key]
    return node


def _evaluate(value: Any, doc: Any) -> Any:
    """Expand ``"=(a/b - c/d)*100"`` against the *uncorrected* document."""
    if isinstance(value, str) and value.startswith("="):
        m = _EXPR.match(value)
        if m is None:
            raise CorrectionError(
                f"unsupported expression {value!r}; only '=(a - b)*100' is allowed")
        left, right = _lookup(doc, m.group("l")), _lookup(doc, m.group("r"))
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise CorrectionError(f"expression {value!r} read a non-number")
        return (left - right) * 100
    if isinstance(value, dict):
        return {k: _evaluate(v, doc) for k, v in value.items()}
    if isinstance(value, list):
        return [_evaluate(v, doc) for v in value]
    return value


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_corrections(path: str | Path | None = None) -> list[dict]:
    """The correction list, validated. Raises on a malformed sidecar."""
    path = Path(path) if path is not None else CORRECTIONS_PATH
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1:
        raise CorrectionError(
            f"{path}: schema_version {doc.get('schema_version')!r}, expected 1")
    out = []
    seen: set[str] = set()
    for i, c in enumerate(doc.get("corrections", [])):
        for field in ("id", "files", "pointer", "op", "reason", "provenance"):
            if field not in c:
                raise CorrectionError(f"{path}: correction {i} has no {field!r}")
        if c["op"] not in _VALID_OPS:
            raise CorrectionError(
                f"{path}: correction {c['id']!r} has op {c['op']!r}, "
                f"expected one of {sorted(_VALID_OPS)}")
        if c["op"] in ("set", "annotate") and "value" not in c:
            raise CorrectionError(
                f"{path}: correction {c['id']!r} is {c['op']!r} but has no value")
        if c["op"] == "annotate" and not isinstance(c["value"], dict):
            raise CorrectionError(
                f"{path}: correction {c['id']!r} is annotate but value is not an object")
        if c["id"] in seen:
            raise CorrectionError(f"{path}: duplicate correction id {c['id']!r}")
        seen.add(c["id"])
        out.append(c)
    return out


def _targets(correction: dict) -> list[Path]:
    return [(REPO_ROOT / f).resolve() for f in correction["files"]]


def _apply(correction: dict, doc: dict, original: dict) -> None:
    op, pointer = correction["op"], correction["pointer"]
    if op == "annotate":
        for container, key in _resolve(doc, _split(pointer)):
            node = container.get(key)
            if not isinstance(node, dict):
                raise CorrectionError(
                    f"{correction['id']}: annotate target {pointer!r} is "
                    f"{type(node).__name__}, not an object")
            node.update(_evaluate(correction["value"], original))
        return
    for container, key in _resolve(doc, _split(pointer)):
        if op == "remove":
            container.pop(key, None)
        else:                                      # set
            container[key] = _evaluate(correction["value"], original)


def load_result(path: str | Path, corrections: list[dict] | None = None,
                strict: bool = True) -> dict:
    """Load a result JSON with its corrections applied in memory.

    The file on disk is never written. The returned dict carries
    ``_corrections_applied``, the ids in the order they were applied, so a
    caller can print provenance beside a table. ``strict=False`` skips a
    correction whose pointer no longer resolves instead of raising, for reading
    a file that has since been re-run.
    """
    path = Path(path)
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    original = json.loads(resolved.read_text(encoding="utf-8"))
    doc = copy.deepcopy(original)
    applied: list[str] = []
    for c in (load_corrections() if corrections is None else corrections):
        if resolved not in _targets(c):
            continue
        try:
            _apply(c, doc, original)
        except CorrectionError:
            if strict:
                raise
            continue
        applied.append(c["id"])
    if applied:
        doc["_corrections_applied"] = applied
        doc["_corrections_source"] = str(
            CORRECTIONS_PATH.relative_to(REPO_ROOT)).replace("\\", "/")
    return doc


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #

def audit(corrections: list[dict] | None = None) -> list[dict]:
    """One row per (correction, file): does it still apply, and did it work?

    A correction whose file has vanished, or whose pointer no longer resolves,
    is reported rather than ignored: it usually means the run was redone and
    the correction is now either obsolete or hiding a live defect.
    """
    rows = []
    for c in (load_corrections() if corrections is None else corrections):
        for f in c["files"]:
            target = (REPO_ROOT / f).resolve()
            row = {"id": c["id"], "file": f, "op": c["op"],
                   "pointer": c["pointer"], "status": "ok", "detail": ""}
            if not target.exists():
                row.update(status="missing-file",
                           detail="named file is not in the repository")
            else:
                try:
                    load_result(target, corrections=[c], strict=True)
                except CorrectionError as e:
                    row.update(status="stale", detail=str(e))
                except (OSError, json.JSONDecodeError) as e:
                    row.update(status="unreadable", detail=f"{type(e).__name__}: {e}")
            rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result", nargs="?", help="a result JSON to print corrected")
    p.add_argument("--audit", action="store_true",
                   help="check every correction against the files it names")
    args = p.parse_args(argv)

    if args.audit or args.result is None:
        rows = audit()
        width = max((len(r["id"]) for r in rows), default=4)
        bad = 0
        for r in rows:
            if r["status"] != "ok":
                bad += 1
            print(f"{r['status']:<14} {r['id']:<{width}}  {r['file']}"
                  + (f"\n{'':<14} {r['detail']}" if r["detail"] else ""))
        print(f"\n{len(rows)} correction/file pairs, {bad} not ok")
        return 1 if bad else 0

    doc = load_result(args.result)
    print(json.dumps(doc, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
