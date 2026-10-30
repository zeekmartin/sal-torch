#!/usr/bin/env python3
"""Aggregate the exp04 SAL+QAT seeds into one table.

Reads every ``seed*/sal_qat_prototype.json`` under a results directory and
prints the per-seed matrices, the across-seed mean, and the key-cell deltas.
Seeds 42/123/456 were also run on NVIDIA for v0.5.0, so for those it prints the
ROCm-vs-NVIDIA comparison that tells us whether the platform changed anything.

    python experiments/aggregate_exp04.py results/lumi_v060/exp04_sal_qat_vitb16
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

CONFIGS = ("fp16", "prune33", "prune50", "int8", "int4", "prune50_int4")
LABELS = {"fp16": "FP16", "prune33": "Prune33%", "prune50": "Prune50%",
          "int8": "INT8", "int4": "INT4", "prune50_int4": "P50+I4"}
ARMS = (("baseline", "Baseline"), ("sal", "SAL only"),
        ("qat", "QAT only"), ("sal_qat", "SAL + QAT"))
KEY = "prune50_int4"

# Published v0.5.0 key-cell deltas (sal_qat - baseline, pp) on NVIDIA.
NVIDIA_KEY_DELTA = {42: 15.51, 123: 25.28, 456: 25.13}


def load(root: Path) -> dict[int, dict]:
    out = {}
    for d in sorted(root.glob("seed*")):
        f = d / "sal_qat_prototype.json"
        if f.exists():
            j = json.loads(f.read_text())
            out[int(j["seed"])] = j
    if not out:
        raise SystemExit(f"no seed*/sal_qat_prototype.json found under {root}")
    return out


def acc(j: dict, arm: str, cfg: str):
    cell = j["results"].get(arm, {}).get(cfg)
    return None if cell is None else cell["accuracy"] * 100


def fmt(v) -> str:
    return "    -   " if v is None else f"{v:6.1f}% "


def table(title: str, get) -> str:
    head = "  {:<11s}".format("") + "".join(f"{LABELS[c]:>8s}" for c in CONFIGS)
    lines = [title, head, "  " + "-" * (11 + 8 * len(CONFIGS))]
    for key, label in ARMS:
        lines.append("  {:<11s}".format(label)
                     + "".join(fmt(get(key, c)) for c in CONFIGS))
    return "\n".join(lines)


def main(argv=None):
    argv = argv or sys.argv[1:]
    root = Path(argv[0] if argv else "results/lumi_v060/exp04_sal_qat_vitb16")
    seeds = load(root)

    first = next(iter(seeds.values()))
    print(f"exp04 — SAL+QAT on {first['model']} / {first['dataset']}")
    print(f"  {len(seeds)} seeds: {sorted(seeds)}")
    print(f"  {first['epochs']} epochs, {first['n_train']} train / {first['n_val']} val, "
          f"batch {first['batch_size']}, mask_ratio {first['mask_ratio']}")
    print(f"  device {first['device']} ({first.get('vram_gb')} GB), "
          f"QAT backend {first['training']['sal_qat'].get('qat_backend')}")
    print()

    for seed in sorted(seeds):
        j = seeds[seed]
        print(table(f"seed {seed}:", lambda a, c, j=j: acc(j, a, c)))
        print()

    print(table(f"MEAN over {len(seeds)} seeds:",
                lambda a, c: statistics.fmean(
                    [v for j in seeds.values() if (v := acc(j, a, c)) is not None])
                if any(acc(j, a, c) is not None for j in seeds.values()) else None))
    print()

    print("Key cell - P50+I4, SAL+QAT vs Baseline:")
    deltas = []
    for seed in sorted(seeds):
        j = seeds[seed]
        b, s = acc(j, "baseline", KEY), acc(j, "sal_qat", KEY)
        if b is None or s is None:
            print(f"  seed {seed:<5d} n/a")
            continue
        d = s - b
        deltas.append(d)
        ref = NVIDIA_KEY_DELTA.get(seed)
        extra = ""
        if ref is not None:
            extra = f"   | NVIDIA v0.5.0: {ref:+.2f}pp   (ROCm - NVIDIA: {d - ref:+.2f}pp)"
        print(f"  seed {seed:<5d} baseline {b:5.1f}%  sal_qat {s:5.1f}%  "
              f"delta {d:+6.2f}pp{extra}")

    if deltas:
        print(f"\n  mean {statistics.fmean(deltas):+.2f}pp over {len(deltas)} seeds"
              + (f", sd {statistics.stdev(deltas):.2f}" if len(deltas) > 1 else ""))
        wins = sum(1 for d in deltas if d > 0)
        print(f"  SAL+QAT beats baseline on {wins}/{len(deltas)} seeds")

        repro = [(s, acc(seeds[s], 'sal_qat', KEY) - acc(seeds[s], 'baseline', KEY)
                  - NVIDIA_KEY_DELTA[s])
                 for s in sorted(seeds) if s in NVIDIA_KEY_DELTA]
        if repro:
            print(f"\n  ROCm vs NVIDIA on the {len(repro)} shared seeds: "
                  f"mean difference {statistics.fmean(d for _, d in repro):+.2f}pp")

    # Also report SAL+QAT against SAL alone — the increment QAT actually adds.
    print("\nIncrement from adding QAT to SAL (P50+I4):")
    incs = []
    for seed in sorted(seeds):
        j = seeds[seed]
        s_only, both = acc(j, "sal", KEY), acc(j, "sal_qat", KEY)
        if s_only is None or both is None:
            continue
        incs.append(both - s_only)
        print(f"  seed {seed:<5d} sal {s_only:5.1f}%  sal_qat {both:5.1f}%  "
              f"delta {both - s_only:+6.2f}pp")
    if incs:
        print(f"  mean {statistics.fmean(incs):+.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
