"""Aggregate I-JEPA ViT-H/14 SAL benchmark runs across seeds.

    python scripts/aggregate_jepa_multiseed.py \
        data/results/jepa_sal_benchmark.json \
        data/results/jepa_sal_benchmark_seed123.json \
        data/results/jepa_sal_benchmark_seed456.json

For each metric x pruning setting, reports SAL and control mean +/- std
(sample std, ddof=1) and how many seeds SAL beat the control. Every metric is
higher-is-better; a tie is not a win.
"""
import argparse
import json
import statistics
from pathlib import Path

SETTINGS = ["random-33%", "random-50%", "magnitude-33%", "magnitude-50%"]
METRICS = {"probe": "linear_probe", "knn": "knn_accuracy", "cka": "cka_similarity"}


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="jepa_sal_benchmark.json per seed")
    p.add_argument("--output", default="data/results/jepa_sal_multiseed_summary.json")
    args = p.parse_args()

    runs = [json.loads(Path(f).read_text()) for f in args.runs]
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds: {seeds}")
    for f, r in zip(args.runs, runs):
        if r.get("smoke") or r.get("control_losses") is None:
            raise SystemExit(f"{f}: smoke run or no --control arm")

    summary, total_wins, total = {}, 0, 0
    for setting in SETTINGS:
        summary[setting] = {}
        for short, key in METRICS.items():
            sal = [r["results"][f"sal+{setting}"][key] for r in runs]
            ctrl = [r["results"][f"ctrl+{setting}"][key] for r in runs]
            wins = sum(s > c for s, c in zip(sal, ctrl))
            sm, ss = mean_std(sal)
            cm, cs = mean_std(ctrl)
            summary[setting][short] = {
                "sal_mean": sm, "sal_std": ss, "ctrl_mean": cm, "ctrl_std": cs,
                "delta_mean": sm - cm, "sal_wins": wins, "n_seeds": len(runs),
                "sal_per_seed": dict(zip(map(str, seeds), sal)),
                "ctrl_per_seed": dict(zip(map(str, seeds), ctrl)),
            }
            total_wins += wins
            total += len(runs)

    per_seed_wins = {
        str(r["seed"]): sum(
            r["results"][f"sal+{s}"][k] > r["results"][f"ctrl+{s}"][k]
            for s in SETTINGS for k in METRICS.values())
        for r in runs
    }
    unpruned = {
        name: {short: mean_std([r["results"][name][key] for r in runs])
               for short, key in METRICS.items()}
        for name in ("sal-trained", "control-trained")
    }

    out = {
        "model": runs[0]["model"], "dataset": runs[0]["dataset"],
        "epochs": runs[0]["epochs"], "mask_ratio": runs[0]["mask_ratio"],
        "seeds": seeds,
        "std": "sample standard deviation (ddof=1) across seeds",
        "win_rule": "SAL wins when sal+X metric > ctrl+X metric (strict); "
                    "all metrics higher-is-better; CKA vs each arm's own unpruned model",
        "settings": summary,
        "unpruned": {n: {m: {"mean": v[0], "std": v[1]} for m, v in d.items()}
                     for n, d in unpruned.items()},
        "wins_per_seed": per_seed_wins,
        "sal_wins_total": total_wins, "comparisons_total": total,
        "source_files": args.runs,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"seeds {seeds}  —  SAL wins {total_wins}/{total}  (per seed: {per_seed_wins})")
    print(f"{'setting':<15}{'metric':<7}{'SAL':>18}{'control':>18}{'delta':>9}{'wins':>7}")
    for setting in SETTINGS:
        for short in METRICS:
            d = summary[setting][short]
            print(f"{setting:<15}{short:<7}"
                  f"{d['sal_mean']:>11.4f} ±{d['sal_std']:.4f}"
                  f"{d['ctrl_mean']:>11.4f} ±{d['ctrl_std']:.4f}"
                  f"{d['delta_mean']:>+9.4f}{d['sal_wins']:>5}/{d['n_seeds']}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
