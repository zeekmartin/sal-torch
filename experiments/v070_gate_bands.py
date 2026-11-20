#!/usr/bin/env python3
"""The v0.7.0 release gate's six bands, pre-registered.

No v0.7.0 tag until the shipped `CompressionPipeline` reproduces archived
cells. Every accuracy this project publishes came from
`scripts/run_sal_qat_prototype.py` and the experiment scripts, not from the
pipeline, which is how a missing recovery stage survived two releases.

**Every cell runs twice**, because `recover()` changes the recipe:

*paper mode* asks the product to reproduce the archived cell under the recipe
that cell was measured with, and is **two-sided**. That is what makes the
published numbers reproducible through the product rather than only through the
harness.

*default mode* asks the product's own defaults to be **no worse** than the
archived cell, and is **one-sided**: accuracy at or above the lower bound for
vision, perplexity at or below the upper bound for the decoder. A changed
recipe that comes in better must pass, not fail.

A two-sided band therefore **fails a result that is better than the archive**,
and that is deliberate: paper mode is a reproduction, and a reproduction that
lands outside the cell it reproduces has not reproduced it, whichever side it
lands on. The one-sided band is where "better is fine" lives. If paper mode
fails high, the thing to do is find out what changed, not widen the band.

For the vision family the two modes are the **same configuration**, and that is
a finding rather than an oversight: the archived vision recipe already clips at
1.0 and already warms up before decaying (linear warmup over 5% of the steps,
then cosine), so the recipe change bites only on the decoder, whose archived
cells ran an unclipped AdamW under a one-cycle schedule. G1 and G2 are
therefore satisfied by one set of three runs each, scored twice.

**Tolerance.** A two-sided per-seed bound is the archived per-seed range
widened by the campaign's own same-seed rerun floor, 0.040 to 0.170pp measured
on four dense-control pairs (`tables/CANONICAL_TABLES.md` section 11); the
widening uses the top of that floor. The decoder has no measured same-seed
rerun pair, so its widening is a **declared** 0.10 perplexity and is marked as
declared, not measured.

Nothing here reads a result file or a GPU: the bands are the pre-registration,
and `check()` is pure. Run this module to print them before any job.

Sources are archive-relative paths in the public data archive,
DOI 10.5281/zenodo.24391200.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

ARCHIVE_DOI = "10.5281/zenodo.24391200"

#: The measured same-seed rerun floor, in points of accuracy, over four
#: dense-control pairs. Used as the two-sided per-seed widening for vision.
RERUN_FLOOR_PP = (0.040, 0.170)

#: No decoder cell was ever re-run at a fixed seed, so this one is declared
#: rather than measured, and says so wherever it is used.
DECLARED_PPL_TOLERANCE = 0.10


@dataclass
class Band:
    """One pre-registered criterion. ``higher_is_better`` picks the direction."""
    name: str
    cell: str
    mode: str                       # "paper" or "default"
    two_sided: bool
    higher_is_better: bool
    metric: str
    mean_lo: Optional[float] = None
    mean_hi: Optional[float] = None
    seed_lo: Optional[float] = None
    seed_hi: Optional[float] = None
    cost_lo: Optional[float] = None
    cost_hi: Optional[float] = None
    archived_seeds: tuple = ()
    archived_mean: Optional[float] = None
    archived_cost: Optional[float] = None
    control: str = ""
    tolerance: str = ""
    sources: tuple = ()
    notes: str = ""
    runs: str = ""                  # which run set satisfies it
    extra: dict = field(default_factory=dict)

    def check(self, per_seed, cost=None) -> dict:
        """Does this run meet the band? Returns the verdict and every number."""
        if not per_seed:
            return {"band": self.name, "passed": False,
                    "why": ["no seeds were scored"]}
        mean = sum(per_seed) / len(per_seed)
        why, ok = [], True

        if self.two_sided:
            if not (self.mean_lo <= mean <= self.mean_hi):
                ok = False
                why.append("mean %.4f outside [%.4f, %.4f]"
                           % (mean, self.mean_lo, self.mean_hi))
            for i, v in enumerate(per_seed):
                if not (self.seed_lo <= v <= self.seed_hi):
                    ok = False
                    why.append("seed %d value %.4f outside [%.4f, %.4f]"
                               % (i, v, self.seed_lo, self.seed_hi))
            if self.cost_lo is not None:
                if cost is None:
                    ok = False
                    why.append("no pruning cost was reported, and this band "
                               "requires one: without a matched control the "
                               "cost is not isolated")
                elif not (self.cost_lo <= cost <= self.cost_hi):
                    ok = False
                    why.append("cost %.4f outside [%.4f, %.4f]"
                               % (cost, self.cost_lo, self.cost_hi))
        elif self.higher_is_better:
            if mean < self.mean_lo:
                ok = False
                why.append("mean %.4f below the archived lower bound %.4f"
                           % (mean, self.mean_lo))
        else:
            if mean > self.mean_hi:
                ok = False
                why.append("mean %.4f above the archived upper bound %.4f"
                           % (mean, self.mean_hi))

        return {"band": self.name, "cell": self.cell, "mode": self.mode,
                "two_sided": self.two_sided, "metric": self.metric,
                "n_seeds": len(per_seed), "per_seed": list(per_seed),
                "mean": mean, "cost": cost, "passed": ok,
                "why": why or ["within the pre-registered band"],
                "archived_mean": self.archived_mean,
                "archived_cost": self.archived_cost,
                "tolerance": self.tolerance, "sources": list(self.sources)}

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def _widen(values, pad):
    return min(values) - pad, max(values) + pad


# --------------------------------------------------------------------- vision
#: ViT-B/16 / ImageNet-1k, 50% of heads, 2 epochs of recovery, 3 seeds,
#: against the 2-epoch dense control (81.554%, 3 seeds, nothing pruned).
_VIT_CONTROL = ("the archived 2-epoch dense control, 81.554% over 3 seeds, "
                "verified against its published mean before anything is "
                "derived from it")
_G1_SEEDS = (78.838, 78.668, 78.942)       # random selection, EXP40B_RANDOM.md
_G1_COST = -2.738
_G2_SEEDS = (77.44, 77.81)                 # magnitude, EXP32_RATIO_AT_2EP.md
_G2_COST = -3.91
_PAD = RERUN_FLOOR_PP[1]

G1_PAPER = Band(
    name="G1-paper", mode="paper", two_sided=True, higher_is_better=True,
    metric="top-1 accuracy, %",
    cell="ViT-B/16 / ImageNet-1k, 50% of heads, random selection, 2 epochs",
    mean_lo=min(_G1_SEEDS), mean_hi=max(_G1_SEEDS),
    seed_lo=_widen(_G1_SEEDS, _PAD)[0], seed_hi=_widen(_G1_SEEDS, _PAD)[1],
    cost_lo=_G1_COST - _PAD, cost_hi=_G1_COST + _PAD,
    archived_seeds=_G1_SEEDS, archived_mean=78.816, archived_cost=_G1_COST,
    control=_VIT_CONTROL,
    tolerance="per-seed range widened by the measured same-seed rerun floor, "
              "%.3fpp" % _PAD,
    sources=("data/oss/results/lumi_v060/EXP40B_RANDOM.md",
             "data/oss/results/lumi_v060/CONTROL_CONVENTION.md"),
    runs="run set A (3 seeds)",
    notes="the shipped default selection, so this is the band the product is "
          "actually judged on")

G1_DEFAULT = Band(
    name="G1-default", mode="default", two_sided=False, higher_is_better=True,
    metric="top-1 accuracy, %", cell=G1_PAPER.cell,
    mean_lo=min(_G1_SEEDS), archived_mean=78.816, control=_VIT_CONTROL,
    tolerance="one-sided: no worse than the archived per-seed minimum",
    sources=G1_PAPER.sources, runs="run set A, scored again",
    notes="satisfied by the same three runs as G1-paper, because the vision "
          "paper recipe and the shipped defaults are the same configuration")

G2_PAPER = Band(
    name="G2-paper", mode="paper", two_sided=True, higher_is_better=True,
    metric="top-1 accuracy, %",
    cell="ViT-B/16 / ImageNet-1k, 50% of heads, magnitude selection, 2 epochs",
    mean_lo=min(_G2_SEEDS), mean_hi=max(_G2_SEEDS),
    seed_lo=_widen(_G2_SEEDS, _PAD)[0], seed_hi=_widen(_G2_SEEDS, _PAD)[1],
    cost_lo=_G2_COST - _PAD, cost_hi=_G2_COST + _PAD,
    archived_seeds=_G2_SEEDS, archived_mean=77.642, archived_cost=_G2_COST,
    control=_VIT_CONTROL,
    tolerance="published three-seed span widened by the measured same-seed "
              "rerun floor, %.3fpp" % _PAD,
    sources=("data/oss/results/lumi_v060/EXP32_RATIO_AT_2EP.md",
             "data/oss/results/lumi_v060/CONTROL_CONVENTION.md"),
    runs="run set B (3 seeds)",
    notes="not the shipped default, but the cell the public ROADMAP "
          "pre-registered, so it is verified rather than quietly replaced")

G2_DEFAULT = Band(
    name="G2-default", mode="default", two_sided=False, higher_is_better=True,
    metric="top-1 accuracy, %", cell=G2_PAPER.cell,
    mean_lo=min(_G2_SEEDS), archived_mean=77.642, control=_VIT_CONTROL,
    tolerance="one-sided: no worse than the published lower bound",
    sources=G2_PAPER.sources, runs="run set B, scored again")

# -------------------------------------------------------------------- decoder
#: OPT-1.3B, `heads_random_50`, 32M C4 tokens, WikiText-2 perplexity, 3 seeds,
#: against the archived dense control (15.2639 over 3 seeds).
_G3_SEEDS = (21.8833, 21.7828, 22.3740)
_G3_COST = 6.7495
_G3_SPREAD = 0.8328

G3_PAPER = Band(
    name="G3-paper", mode="paper", two_sided=True, higher_is_better=False,
    metric="WikiText-2 perplexity",
    cell="OPT-1.3B, heads_random_50, 32M C4 tokens, published recipe "
         "(unclipped AdamW, one_cycle, pct_start 0.10)",
    mean_lo=min(_G3_SEEDS), mean_hi=max(_G3_SEEDS),
    seed_lo=min(_G3_SEEDS) - DECLARED_PPL_TOLERANCE,
    seed_hi=max(_G3_SEEDS) + DECLARED_PPL_TOLERANCE,
    cost_lo=_G3_COST - _G3_SPREAD / 2, cost_hi=_G3_COST + _G3_SPREAD / 2,
    archived_seeds=_G3_SEEDS, archived_mean=22.0134, archived_cost=_G3_COST,
    control="the archived dense control, 15.2639 over 3 seeds, after the same "
            "32M C4 tokens",
    tolerance="per-seed range widened by a DECLARED %.2f perplexity: no "
              "decoder cell was ever re-run at a fixed seed, so there is no "
              "measured floor here" % DECLARED_PPL_TOLERANCE,
    sources=("data/pro/results/EXP36_OPT_SPARSE.md",
             "data/pro/results/exp36_rec_*/exp36_opt_sparse.json"),
    runs="run set C (3 seeds)",
    notes="run with max_grad_norm=None and schedule='one_cycle', which exist "
          "in recover() so that a published cell can be reproduced exactly as "
          "it was measured. They are not recommended settings")

G3_DEFAULT = Band(
    name="G3-default", mode="default", two_sided=False, higher_is_better=False,
    metric="WikiText-2 perplexity",
    cell="OPT-1.3B, heads_random_50, 32M C4 tokens, shipped defaults "
         "(max_grad_norm=1.0, warmup_cosine)",
    mean_hi=max(_G3_SEEDS), archived_mean=22.0134,
    control=G3_PAPER.control,
    tolerance="one-sided: no worse than the archived per-seed maximum. This "
              "is the only family where the recipe actually changed, so it is "
              "the only place the two modes measure different things",
    sources=G3_PAPER.sources, runs="run set D (3 seeds)")

BANDS = {b.name: b for b in (G1_PAPER, G1_DEFAULT, G2_PAPER, G2_DEFAULT,
                             G3_PAPER, G3_DEFAULT)}

#: Which run set each band is scored from. Four sets, six bands.
RUN_SETS = {
    "A": ("G1-paper", "G1-default"),
    "B": ("G2-paper", "G2-default"),
    "C": ("G3-paper",),
    "D": ("G3-default",),
}


def check(name: str, per_seed, cost=None) -> dict:
    if name not in BANDS:
        raise KeyError("unknown band %r; have %s" % (name, sorted(BANDS)))
    return BANDS[name].check(per_seed, cost)


def check_all(results: dict) -> dict:
    """``{band name: {"per_seed": [...], "cost": x}}`` to one verdict."""
    verdicts = {n: check(n, r.get("per_seed", []), r.get("cost"))
                for n, r in results.items()}
    return {"bands": verdicts,
            "passed": all(v["passed"] for v in verdicts.values()),
            "n_checked": len(verdicts), "n_bands": len(BANDS),
            "complete": set(verdicts) == set(BANDS)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="print as JSON")
    args = p.parse_args(argv)

    if args.json:
        print(json.dumps({n: b.to_dict() for n, b in BANDS.items()}, indent=1,
                         default=list))
        return 0

    print("v0.7.0 release gate: six pre-registered bands")
    print("archive DOI %s\n" % ARCHIVE_DOI)
    for name, b in BANDS.items():
        print("%s  [%s, %s]" % (name, b.mode,
                                "two-sided" if b.two_sided else "one-sided"))
        print("  cell:      %s" % b.cell)
        print("  metric:    %s" % b.metric)
        if b.two_sided:
            print("  mean in    [%.4f, %.4f]" % (b.mean_lo, b.mean_hi))
            print("  each seed  [%.4f, %.4f]" % (b.seed_lo, b.seed_hi))
            if b.cost_lo is not None:
                print("  cost in    [%.4f, %.4f]" % (b.cost_lo, b.cost_hi))
        elif b.higher_is_better:
            print("  mean       >= %.4f" % b.mean_lo)
        else:
            print("  mean       <= %.4f" % b.mean_hi)
        print("  control:   %s" % b.control)
        print("  tolerance: %s" % b.tolerance)
        print("  runs:      %s" % b.runs)
        if b.notes:
            print("  note:      %s" % b.notes)
        print("  sources:   %s" % ", ".join(b.sources))
        print()
    print("four run sets satisfy six bands: %s" % RUN_SETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
