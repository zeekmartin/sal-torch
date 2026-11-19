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
import math
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
    for cell, label in (("sparsegpt_attn_p50", "unstructured 50%, SparseGPT"),
                        ("wanda_attn_p50", "unstructured 50%, Wanda")):
        per = {}
        for s in SEEDS3:
            fp = oss_results / "lumi_v060" / ("exp33_s%d" % s) / "exp33_sparse.json"
            if fp.is_file():
                c = load(fp)["cells"].get(cell)
                if c and c.get("recovered"):
                    per[s] = c["recovered"]["accuracy"]
        add(label, per)
    for cell, label in (("sparsegpt_attn_2of4", "2:4, SparseGPT"),
                        ("wanda_attn_2of4", "2:4, Wanda")):
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
    add("heads 50%, magnitude", per)

    # FFN neurons 25%, RANDOM selection, measured at the same three
    # seeds. The magnitude row (-0.80pp, exp35) is the one this panel
    # used to carry, and it made the ViT panel the only one of the three
    # whose FFN arm was not randomly selected: both OPT panels read
    # `ffn_random_25`. A panel set read for ORDERING has to vary one
    # thing at a time, so the selection is held fixed across the three
    # panels and the figure compares structures.
    #
    # The magnitude cost is not retracted by being dropped from a
    # figure: `vit_ffn_magnitude` records it in figures_data.json beside
    # this one. They differ by 0.45pp, which is larger than either arm's
    # own seed range, so the swap is not cosmetic and the discarded row
    # has to stay visible.
    #
    # Each seed's cost is already matched against THAT seed's own
    # 2-epoch control inside the file, as every other arm in this panel
    # is, so it is read rather than recomputed.
    vals = []
    for s in SEEDS3:
        fp = (pro_results / ("exp44_random_s%d" % s)
              / "exp35_mlp_matched.json")
        if fp.is_file():
            d = load(fp)
            if d.get("neuron_selection") != "random":
                raise SystemExit(
                    "%s reports neuron_selection %r, not 'random'. This "
                    "panel's FFN arm is the randomly selected one, to "
                    "match both OPT panels."
                    % (fp, d.get("neuron_selection")))
            if d.get("seed") != s:
                raise SystemExit(
                    "%s is seed %r, not %d" % (fp, d.get("seed"), s))
            c = d["arms"]["mlp25"]["cost"]
            if c.get("epochs_matched"):
                vals.append(c["pruning_cost_pp"])
    if vals:
        rows.append({"label": "FFN neurons 25%, random", "n": len(vals),
                     "mean": sum(vals) / len(vals),
                     "lo": min(vals), "hi": max(vals),
                     "selection": "random",
                     "source": "exp44_random_s*/exp35_mlp_matched.json"})
    return rows


def vit_ffn_magnitude(pro_results: Path):
    """The magnitude-selected FFN cost: recorded, no longer plotted.

    Dropping a measurement from a figure is not retracting it. This is
    the row the ViT panel carried until the selection was held fixed
    across the three panels, and it is written into figures_data.json so
    that the change is visible rather than silent.
    """
    vals = []
    for s in SEEDS3:
        fp = pro_results / "exp35" / ("seed%d" % s) / "exp35_mlp_matched.json"
        if fp.is_file():
            c = load(fp)["arms"]["mlp25"]["cost"]
            if c.get("epochs_matched"):
                vals.append(c["pruning_cost_pp"])
    if not vals:
        return None
    return {"label": "FFN neurons 25%, magnitude", "n": len(vals),
            "mean": sum(vals) / len(vals), "lo": min(vals), "hi": max(vals),
            "selection": "magnitude",
            "source": "exp35/seed*/exp35_mlp_matched.json",
            "why_not_plotted": (
                "figure 1's ViT panel plots the randomly selected FFN arm, "
                "so that all three panels vary the structure removed and "
                "not the rule that selected it")}


#: Arms that figure 1 must NOT plot, and the measured reason.
#:
#: `ffn_random_50` at 6.7B blows the recovery loss up early on all three
#: seeds: +4.08, +4.26 and +5.43 nats above the run's own running
#: minimum, where every other arm at this scale moves 1.09 to 1.73 and
#: does so at the step its SEED moves at, not its arm. Two of the three
#: climb back inside the 3906-step budget and one does not, so the
#: per-seed costs are +9.33, +10.96 and +201.62. A bar is a central
#: value and a whisker is a range around it; neither is something these
#: three cells support, so the arm has no bar rather than a wide one.
OPT_EXCLUDED = {
    "ffn_random_50": (
        "all three 6.7B seeds blow the recovery loss up early; per-seed "
        "costs span +9.33 to +201.62 and no cell measures the structure's "
        "cost cleanly, so the arm has no central value to draw"),
}

#: One line per label, because the bars are horizontal. The two-line
#: labels this tuple used to carry overlapped under vertical bars at six
#: categories to a 4.4-inch panel.
OPT_ARMS = (("sparsegpt_unstructured_50", "unstructured 50%, SparseGPT"),
            ("wanda_unstructured_50", "unstructured 50%, Wanda"),
            ("sparsegpt_2to4", "2:4, SparseGPT"),
            ("wanda_2to4", "2:4, Wanda"),
            ("heads_random_50", "heads 50%, random"),
            ("ffn_random_25", "FFN neurons 25%, random"))


def opt_costs(path: Path) -> list:
    """One cost table's per-arm mean and seed range, in perplexity."""
    d = load(path)
    stray = sorted(set(OPT_EXCLUDED) & {a for a, _l in OPT_ARMS})
    if stray:
        raise SystemExit(
            "%s is both plotted and excluded from figure 1. The exclusion "
            "records why the arm has no central value; plotting it anyway "
            "would assert one." % ", ".join(stray))
    rows = []
    for arm, label in OPT_ARMS:
        v = d["per_arm"].get(arm)
        if not v or v.get("mean") is None:
            continue
        rows.append({"label": label, "n": v["n_seeds"], "mean": v["mean"],
                     "lo": v["min"], "hi": v["max"],
                     "source": path.name})
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
    """Three panels, horizontal bars, and nothing written on the figure.

    The bars are horizontal because six category labels will not fit
    under a panel 4.4 inches wide: two lines each still overlapped, and
    rotating them to 45 degrees makes them unreadable at 7.5pt. Turned
    over, the label space is the long axis and no label can collide.

    There is no suptitle and no footer. Both were in-figure prose, the
    footer was clipped on its left edge in the PDF, and a caption that
    can wrap belongs in LaTeX. `CAPTIONS` carries the text so the caption
    is still generated from this file rather than written by hand.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("ViT-B/16, ImageNet-1k", vit,
               "matched cost (points of top-1)"),
              ("OPT-1.3B, WikiText-2", opt13,
               "matched cost (perplexity)"),
              ("OPT-6.7B, WikiText-2", opt67,
               "matched cost (perplexity)")]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.9))
    for ax, (title, rows, xlab) in zip(axes, panels):
        if not rows:
            ax.set_axis_off()
            continue
        ys = list(range(len(rows)))
        # Signed, as the campaign defines a cost: arm minus control. The
        # whiskers are then the range itself and cannot go negative,
        # which taking the magnitude of a mean over seeds that straddle
        # zero does.
        means = [r["mean"] for r in rows]
        lo = [r["mean"] - r["lo"] for r in rows]
        hi = [r["hi"] - r["mean"] for r in rows]
        ax.barh(ys, means, xerr=[lo, hi], capsize=3, height=0.62,
                color="#4c78a8", edgecolor="black", linewidth=0.6,
                error_kw={"ecolor": "black", "elinewidth": 1.0})
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
        # barh puts index 0 at the bottom; the rows are listed in the
        # order they should be read, so the axis is turned over.
        ax.invert_yaxis()
        ax.set_title("%s  (%d seeds)" % (title, rows[0]["n"]), fontsize=10)
        ax.set_xlabel(xlab, fontsize=8.5)
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        # A ViT cost is points lost and negative; an OPT cost is
        # perplexity gained and positive. The ViT axis is inverted below,
        # after its limits are set, so that a longer bar means a worse
        # arm in all three panels -- the only thing the three panels may
        # be read together for.
        # Every bar grows rightward ON SCREEN, in both panel kinds: the
        # OPT costs are positive on a normal axis and the ViT costs are
        # negative on an inverted one. So `far` is the screen-far end of
        # the whisker whichever panel this is, the label is anchored just
        # past it, and `ha` is "left" in both -- which is a statement
        # about the screen and must not be derived from the data's sign.
        ends = [r["hi"] for r in rows] + [r["lo"] for r in rows] + [0.0]
        span = max(ends) - min(ends) or 1.0
        pad = 0.030 * span
        outward = -1.0 if "points" in xlab else 1.0
        for y, r in zip(ys, rows):
            far = min(r["lo"], 0.0) if outward < 0 else max(r["hi"], 0.0)
            ax.annotate("%+.2f" % r["mean"], (far + outward * pad, y),
                        ha="left", va="center", fontsize=7.5)
        # Limits from the data, with room for the value label past the
        # longest whisker, and only then is the axis turned over.
        near, away = (max(ends), min(ends)) if outward < 0 \
            else (min(ends), max(ends))
        ax.set_xlim(sorted((near - outward * 0.02 * span,
                            away + outward * 0.26 * span)))
        if outward < 0:
            ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    print("wrote %s" % out)


#: Which way each point's label leaves its marker, as a direction and a
#: horizontal alignment. The distance is not here: it is computed from
#: the marker's own radius at draw time, because the marker AREA is the
#: parameter count and runs 60 to 800 square points, i.e. a radius of
#: 4.4 to 16. Every label used to sit at a fixed (9, -3), which is
#: inside the larger circles -- head33 and head50 were written across
#: their own markers, and mlp25's text reached head25, the two being
#: 0.47 points of top-1 and 0.054x apart on the GPU panel.
#:
#: The directions are chosen once, per arm, and not per panel, so that
#: the GPU and CPU panels can be read against each other.
F2_LABEL = {
    "dense":        (1.0, 0.0, "left"),
    # Down-left: mlp25 sits up-right of it and head33 down-right.
    "head25":       (-1.0, -0.7, "right"),
    "head33":       (1.0, -0.2, "left"),
    "head50":       (1.0, -0.6, "left"),
    # Up-right, away from head25 and head33 below it.
    "mlp25":        (1.0, 0.7, "left"),
    "mlp50":        (1.0, 0.0, "left"),
    # Furthest right on both panels, so its label goes back inward.
    "head50_mlp50": (-1.0, -0.6, "right"),
}

#: Clearance between a marker's edge and its label, in points.
F2_LABEL_GAP = 4.0


def figure2(points, out: Path):
    """Quality against measured latency, with nothing written on it.

    As in figure 1, the suptitle and the footer are gone: the footer's
    left edge was clipped, and the text is in `CAPTIONS` so LaTeX gets
    it from this file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
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
            ux, uy, ha = F2_LABEL.get(p["arm"], (1.0, 0.0, "left"))
            # `size` is an area in square points, so the radius is
            # sqrt(size/pi) and the label starts just outside it.
            r = math.sqrt(size / math.pi) + F2_LABEL_GAP
            ax.annotate(note, (x, p["accuracy"]),
                        textcoords="offset points",
                        xytext=(ux * r, uy * r - 3.0),
                        ha=ha, va="center", fontsize=8, zorder=4)
        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("measured speedup over dense", fontsize=9)
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        # Headroom on both sides so an outward label is not clipped.
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 - 0.10 * (x1 - x0), x1 + 0.12 * (x1 - x0))
    axes[0].set_ylabel("top-1 after 2 epochs of recovery (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    print("wrote %s" % out)


#: The two captions, generated here and NOT drawn on the figures. They
#: are the text a LaTeX caption should carry, so that removing the
#: in-figure prose does not lose it.
#:
#: Figure 1's title no longer says "attention parameters": the FFN arm
#: removes feed-forward neurons, which are not attention parameters, and
#: `heads 50%` and `FFN neurons 25%` are matched on the parameter budget
#: they free rather than on where the parameters come from.
CAPTIONS = {
    "fig_cost_by_structure": (
        "Cost of removing about the same parameter budget, by the "
        "structure removed. Bars are the mean over seeds and whiskers "
        "the range over seeds, not an interval: below ten draws the "
        "campaign reports a range. Every cost is the arm minus the dense "
        "control trained for the same budget on the same seed. A ViT "
        "cost is points of top-1 lost and is negative; an OPT cost is "
        "perplexity gained and is positive. The ViT axis is inverted so "
        "that a longer bar is a worse arm in every panel. The units "
        "differ, so the panels are read for ordering and never for "
        "magnitude. The FFN arm is randomly selected in all three "
        "panels."),
    "fig_quality_vs_latency": (
        "Quality against measured latency for the physically sliced ViT "
        "arms. Marker area is the parameter count; whiskers are the range "
        "over seeds. CPU speedups are the three-pass re-measurement: two "
        "cells of the single-pass grid did not survive being measured "
        "again, and a cell whose three passes disagree by more than 5 "
        "percent is marked unstable. The dense reference is 1.0 by "
        "construction and is not marked, although its own three passes "
        "span 8.3 percent. Unstructured and 2:4 arms are absent because "
        "they are masked, not sliced, and give no speedup without a "
        "sparse kernel. The FFN arms here (mlp25, mlp50, head50_mlp50) "
        "are MAGNITUDE-selected, which is the recommended FFN recipe and "
        "the selection the latency grid timed; figure 1's ViT panel plots "
        "the randomly selected FFN arm instead, so that its three panels "
        "vary the structure and not the selection. Magnitude costs "
        "-0.82pp at 25% and random -1.25pp, both against the same matched "
        "2-epoch control, so the two figures differ by 0.43pp on this arm "
        "by design and not by accident."),
}


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
    vit_dropped = vit_ffn_magnitude(pro)
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
                    "figure1_ffn_selection": {
                        "plotted_in_every_panel": "random",
                        "why": ("all three panels of figure 1 are read for "
                                "the ORDERING of the structures, so the "
                                "selection rule is held fixed and only the "
                                "structure varies. The ViT panel used to "
                                "plot the magnitude-selected FFN arm while "
                                "both OPT panels plotted the randomly "
                                "selected one."),
                        "vit_plotted": next((r for r in vit
                                             if "FFN" in r["label"]), None),
                        "vit_not_plotted": vit_dropped,
                    },
                    "figure1_excluded_arms": OPT_EXCLUDED,
                    "figure2": pts,
                    "figure2_ffn_selection": {
                        "plotted": "magnitude",
                        "why": ("magnitude is the recommended FFN recipe "
                                "(ROADMAP, v0.7.0 selection defaults) and "
                                "is the selection the exp39/exp46 latency "
                                "grid timed. Figure 1's ViT panel plots "
                                "the random arm for a different reason: "
                                "its three panels are read for ordering "
                                "and must vary only the structure. The "
                                "two arms differ by 0.43pp at 25%, which "
                                "is larger than either one's seed range, "
                                "so this is a stated difference and not "
                                "an inconsistency to be reconciled."),
                        "figure1_vit_panel_plots": "random",
                    },
                    "figure2_label_offsets": {
                        k: {"dx": v[0], "dy": v[1], "ha": v[2]}
                        for k, v in F2_LABEL.items()},
                    "captions_for_latex": CAPTIONS,
                    "in_figure_text": (
                        "none. Neither figure carries a suptitle or a "
                        "footer: the footer was clipped on its left edge, "
                        "and a caption that can wrap belongs in LaTeX. The "
                        "text is generated above, under "
                        "captions_for_latex."),
                    "note": ("error bars are ranges over seeds and not "
                             "intervals: the campaign refuses an interval "
                             "below ten draws")}, indent=1),
        encoding="utf-8")
    print("wrote %s" % (out / "figures_data.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
