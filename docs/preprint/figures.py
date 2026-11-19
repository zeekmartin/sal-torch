#!/usr/bin/env python3
"""The preprint's two figures, generated from the committed result JSONs.

Nothing here is typed in. Every point is read from a result file, every
error bar is the seed range the campaign's ten-draw rule allows below ten
seeds, and every ViT cost comes through ``matched_control``, which checks
the control ladder against its published mean before deriving anything
and refuses a cost when the recovery lengths differ.

  figure 1  cost by structure at matched parameters, three panels:
            ViT-B/16, OPT-1.3B, OPT-6.7B. Error bars are the range over
            seeds, not an interval: three draws make no coverage claim.
            The two units are different and the panels say so -- ViT is
            accuracy in points, OPT is perplexity -- so the panels are
            read side by side for ORDERING and never for magnitude.

  figure 2  quality against MEASURED latency at batch 32, GPU and CPU,
            for the physically sliced ViT arms. Marker area is the
            parameter count. The CPU speedups are the three-pass
            re-measurement, not the single pass: two of the single-pass
            cells did not survive being measured again.

No em-dash appears in any label. The output is PDF.

    python docs/preprint/figures.py --pro-results <dir> --out <dir>

``--pro-results`` is required rather than defaulted, so this file names
no location outside its own repository. Those result files are deposited
with the preprint in its public data archive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OSS = HERE.parent.parent
sys.path.insert(0, str(OSS / "experiments"))

SEEDS3 = (42, 123, 456)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------- figure 1 data
def vit_costs(oss_results: Path, pro_results: Path) -> list:
    """ViT-B/16 matched costs at two epochs, per seed, by structure.

    Every cost is this seed's arm against this seed's own 2-epoch dense
    control, through matched_control.Control.cost, which emits
    pruning_cost_pp only when the epoch counts agree.
    """
    import matched_control as mc

    ladder = mc.ControlLadder.load((2,), verify=True, root=oss_results.parent)
    rows = []

    def add(label, per_seed):
        vals = []
        for seed, acc in per_seed.items():
            ctl = ladder.per_seed.get((2, seed))
            if ctl is None:
                continue
            r = ctl.cost(acc, 2, label=label)
            if not r.get("epochs_matched"):
                continue
            vals.append(r["pruning_cost_pp"])
        if vals:
            rows.append({"label": label, "n": len(vals),
                         "mean": sum(vals) / len(vals),
                         "lo": min(vals), "hi": max(vals)})

    # unstructured 50% and 2:4, both pruners, from exp33
    # The ATTENTION scope, not the whole-model one: this figure is about
    # half of the attention parameters, and so is the preprint's
    # structure table, whose 77.21% one-shot is this cell's.
    for cell, label in (("sparsegpt_attn_p50", "unstructured 50%\nSparseGPT"),
                        ("wanda_attn_p50", "unstructured 50%\nWanda")):
        per = {}
        for s in SEEDS3:
            fp = oss_results / "lumi_v060" / ("exp33_s%d" % s) / "exp33_sparse.json"
            if fp.is_file():
                c = load(fp)["cells"].get(cell)
                if c and c.get("recovered"):
                    per[s] = c["recovered"]["accuracy"]
        add(label, per)
    for cell, label in (("sparsegpt_attn_2of4", "2:4\nSparseGPT"),
                        ("wanda_attn_2of4", "2:4\nWanda")):
        per = {}
        for s in SEEDS3:
            fp = (oss_results / "lumi_v060" / ("exp33_nm_s%d" % s)
                  / "exp33_sparse.json")
            if fp.is_file():
                c = load(fp)["cells"].get(cell)
                if c and c.get("recovered"):
                    per[s] = c["recovered"]["accuracy"]
        add(label, per)

    # whole heads 50%, from the ratio sweep
    per = {}
    for s in SEEDS3:
        fp = (oss_results / "lumi_v060" / ("exp32_ratios_s%d" % s)
              / "exp29_recovery.json")
        if fp.is_file():
            r = load(fp)["ratios"].get("0.50")
            if r:
                per[s] = r["accuracy"]
    add("heads 50%\nmagnitude", per)

    # FFN neurons 25%, which the pro repository measured. Its own cost
    # row is already matched per seed, so it is read rather than redone.
    vals = []
    for s in SEEDS3:
        fp = pro_results / "exp35" / ("seed%d" % s) / "exp35_mlp_matched.json"
        if fp.is_file():
            c = load(fp)["arms"]["mlp25"]["cost"]
            if c.get("epochs_matched"):
                vals.append(c["pruning_cost_pp"])
    if vals:
        rows.append({"label": "FFN neurons 25%\nmagnitude", "n": len(vals),
                     "mean": sum(vals) / len(vals),
                     "lo": min(vals), "hi": max(vals)})
    return rows


OPT_ARMS = (("sparsegpt_unstructured_50", "unstructured 50%\nSparseGPT"),
            ("wanda_unstructured_50", "unstructured 50%\nWanda"),
            ("sparsegpt_2to4", "2:4\nSparseGPT"),
            ("wanda_2to4", "2:4\nWanda"),
            ("heads_random_50", "heads 50%\nrandom"),
            ("ffn_random_25", "FFN neurons 25%\nrandom"))


def opt_costs(path: Path) -> list:
    """One cost table's per-arm mean and seed range, in perplexity."""
    d = load(path)
    rows = []
    for arm, label in OPT_ARMS:
        v = d["per_arm"].get(arm)
        if not v or v.get("mean") is None:
            continue
        rows.append({"label": label, "n": v["n_seeds"], "mean": v["mean"],
                     "lo": v["min"], "hi": v["max"]})
    return rows


# ------------------------------------------------------------- figure 2 data
#: The sliced ViT arms, with where each one's recovered accuracy comes
#: from. head* are the ratio sweep's own 2-epoch rows; mlp* and the
#: joint arm are the pro repository's matched FFN runs. Both are two
#: epochs of recovery, which is what makes them one axis.
F2_ARMS = ("dense", "head25", "head33", "head50", "mlp25", "mlp50",
           "head50_mlp50")


def latency_points(oss_results: Path, pro_results: Path) -> list:
    e39 = load(pro_results / "exp39" / "exp39_latency.json")
    e46 = load(pro_results / "exp46" / "exp46_cpu_repeat.json")
    cpu3 = e46["speedup_vs_dense"]

    # accuracies, per arm, averaged over the seeds each source has
    acc = {}
    rng = {}
    for ratio, name in (("0.25", "head25"), ("0.33", "head33"),
                        ("0.50", "head50")):
        vals = []
        for s in SEEDS3:
            fp = (oss_results / "lumi_v060" / ("exp32_ratios_s%d" % s)
                  / "exp29_recovery.json")
            if fp.is_file():
                r = load(fp)["ratios"].get(ratio)
                if r:
                    vals.append(r["accuracy"] * 100)
        if vals:
            acc[name] = sum(vals) / len(vals)
            rng[name] = (min(vals), max(vals))
    for armkey, name in (("mlp25", "mlp25"), ("mlp50", "mlp50"),
                         ("head50_mlp50", "head50_mlp50")):
        vals = []
        for s in SEEDS3:
            fp = pro_results / "exp35" / ("seed%d" % s) / "exp35_mlp_matched.json"
            if fp.is_file():
                c = load(fp)["arms"].get(armkey, {}).get("cost")
                if c and c.get("accuracy") is not None:
                    vals.append(c["accuracy"] * 100)
        if vals:
            acc[name] = sum(vals) / len(vals)
            rng[name] = (min(vals), max(vals))
    # The dense point is the 2-epoch control, which is what every arm
    # above is costed against.
    import matched_control as mc
    ladder = mc.ControlLadder.load((2,), verify=True, root=oss_results.parent)
    acc["dense"] = ladder.controls[2].accuracy * 100
    ps = list(ladder.controls[2].per_seed.values())
    rng["dense"] = (min(ps) * 100, max(ps) * 100) if ps else (acc["dense"],
                                                              acc["dense"])

    out = []
    for name in F2_ARMS:
        a = e39["arms"].get(name)
        if a is None or name not in acc:
            continue
        gpu = a["latency"].get("cuda", {}).get("32", {}).get(
            "speedup_vs_dense")
        cpu = (cpu3.get(name) or {}).get("32", {}).get("speedup")
        unstable = (cpu3.get(name) or {}).get("32", {}).get("unstable")
        out.append({"arm": name, "accuracy": acc[name],
                    "acc_lo": rng[name][0], "acc_hi": rng[name][1],
                    "gpu32": gpu, "cpu32": cpu,
                    # The dense speedup is 1.0 by construction, so the
                    # instability of its own timing is not a doubt about
                    # this point. Recorded, not plotted.
                    "cpu_unstable": bool(unstable) and name != "dense",
                    "cpu_pass_spread_pct": (cpu3.get(name) or {})
                                           .get("32", {}).get("spread_pct"),
                    "params": a["params_after"],
                    "param_reduction": a.get("param_reduction")})
    return out


# ------------------------------------------------------------------ drawing
def figure1(vit, opt13, opt67, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("ViT-B/16, ImageNet-1k", vit,
               "matched cost (points of top-1)"),
              ("OPT-1.3B, WikiText-2", opt13,
               "matched cost (perplexity)"),
              ("OPT-6.7B, WikiText-2", opt67,
               "matched cost (perplexity)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, (title, rows, ylab) in zip(axes, panels):
        if not rows:
            ax.set_axis_off()
            continue
        xs = range(len(rows))
        # Signed, as the campaign defines a cost: arm minus control. The
        # whiskers are then the range itself and cannot go negative,
        # which taking the magnitude of a mean over seeds that straddle
        # zero does.
        means = [r["mean"] for r in rows]
        lo = [r["mean"] - r["lo"] for r in rows]
        hi = [r["hi"] - r["mean"] for r in rows]
        ax.bar(xs, means, yerr=[lo, hi], capsize=4, width=0.62,
               color="#4c78a8", edgecolor="black", linewidth=0.6,
               error_kw={"ecolor": "black", "elinewidth": 1.0})
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["label"] for r in rows], fontsize=7.5)
        ax.set_title("%s  (%d seeds)" % (title, rows[0]["n"]), fontsize=10)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        # ViT costs are negative and OPT costs positive, so the ViT axis
        # is inverted: worse is upward in every panel.
        if "points" in ylab:
            ax.invert_yaxis()
        for x, r in zip(xs, rows):
            far = r["lo"] if abs(r["lo"]) > abs(r["hi"]) else r["hi"]
            ax.annotate("%+.2f" % r["mean"], (x, far),
                        textcoords="offset points",
                        xytext=(0, -11 if r["mean"] < 0 else 4),
                        ha="center", fontsize=7)
    fig.suptitle("Cost of removing about half of the attention parameters, "
                 "by the structure removed. Bars are the mean over seeds "
                 "and whiskers the range over seeds, not an interval.",
                 fontsize=9.5, y=0.99)
    fig.text(0.5, 0.005,
             "Every cost is the arm minus the dense control trained for "
             "the same budget on the same seed. A ViT cost is points of "
             "accuracy lost and is negative; an OPT cost is perplexity "
             "gained and is positive. The ViT axis is inverted so that "
             "worse is upward in every panel. The units differ, so the "
             "panels are compared for ordering and not for magnitude.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out, format="pdf")
    print("wrote %s" % out)


def figure2(points, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    pmax = max(p["params"] for p in points)
    for ax, key, title in ((axes[0], "gpu32",
                            "GPU, MI250X one GCD, batch 32"),
                           (axes[1], "cpu32",
                            "CPU, EPYC 7A53, 7 threads, batch 32")):
        for p in points:
            x = p[key]
            if x is None:
                continue
            size = 60 + 740 * (p["params"] / pmax) ** 2
            ax.scatter([x], [p["accuracy"]], s=size, alpha=0.75,
                       color="#e45756" if p["arm"] == "dense" else "#4c78a8",
                       edgecolor="black", linewidth=0.7, zorder=3)
            ax.errorbar([x], [p["accuracy"]],
                        yerr=[[p["accuracy"] - p["acc_lo"]],
                              [p["acc_hi"] - p["accuracy"]]],
                        fmt="none", ecolor="black", elinewidth=0.9,
                        capsize=3, zorder=2)
            note = p["arm"]
            if key == "cpu32" and p["cpu_unstable"]:
                note += " (unstable)"
            ax.annotate(note, (x, p["accuracy"]),
                        textcoords="offset points", xytext=(9, -3),
                        fontsize=8)
        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("measured speedup over dense", fontsize=9)
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("top-1 after 2 epochs of recovery (%)", fontsize=9)
    fig.suptitle("Quality against measured latency for the physically "
                 "sliced arms. Marker area is the parameter count; "
                 "whiskers are the range over seeds.", fontsize=9.5,
                 y=0.98)
    fig.text(0.5, 0.005,
             "CPU speedups are the three-pass re-measurement: two cells "
             "of the single-pass grid did not survive being measured "
             "again, and a cell whose three passes disagree by more than "
             "5 percent is marked unstable. The dense reference is 1.0 by "
             "construction and is not marked, although its own three "
             "passes span 8.3 percent. Unstructured and 2:4 arms are "
             "absent because they are masked, not sliced, and give no "
             "speedup without a sparse kernel.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out, format="pdf")
    print("wrote %s" % out)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pro-results", required=True,
                   help="directory holding exp35/, exp39/, exp46/ and the "
                        "two OPT cost tables")
    p.add_argument("--oss-results", default=str(OSS / "results"))
    p.add_argument("--out", default=str(HERE))
    a = p.parse_args(argv)

    pro, oss = Path(a.pro_results), Path(a.oss_results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    vit = vit_costs(oss, pro)
    opt13 = opt_costs(pro / "exp36_cost_table.json")
    opt67 = opt_costs(pro / "exp36_opt67_cost_table.json")
    for name, rows in (("ViT-B/16", vit), ("OPT-1.3B", opt13),
                       ("OPT-6.7B", opt67)):
        print("%-10s %d arm(s): %s" % (name, len(rows),
                                       ", ".join(r["label"].replace("\n", " ")
                                                 for r in rows)))
    figure1(vit, opt13, opt67, out / "fig_cost_by_structure.pdf")

    pts = latency_points(oss, pro)
    print("latency: %d point(s): %s"
          % (len(pts), ", ".join(x["arm"] for x in pts)))
    figure2(pts, out / "fig_quality_vs_latency.pdf")

    # The numbers behind both figures, written beside them so a reader
    # can check a point without rerunning this.
    (out / "figures_data.json").write_text(
        json.dumps({"generated_by": "docs/preprint/figures.py",
                    "figure1": {"vit": vit, "opt13b": opt13,
                                "opt67b": opt67},
                    "figure2": pts,
                    "note": ("error bars are ranges over seeds and not "
                             "intervals: the campaign refuses an interval "
                             "below ten draws")}, indent=1),
        encoding="utf-8")
    print("wrote %s" % (out / "figures_data.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
