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
    #: ``{seed: archived value}``. The tuple above is an ordered summary and
    #: is NOT safe to zip against a run's values: cells are read in whatever
    #: order the file system hands them over, so seed 123's result would be
    #: compared against seed 42's archive. Anything per-seed uses this map.
    archived_by_seed: dict = field(default_factory=dict)
    archived_mean: Optional[float] = None
    archived_cost: Optional[float] = None
    control: str = ""
    tolerance: str = ""
    sources: tuple = ()
    notes: str = ""
    runs: str = ""                  # which run set satisfies it
    extra: dict = field(default_factory=dict)

    def check(self, per_seed, cost=None, seeds=None) -> dict:
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
                "seeds": list(seeds) if seeds else [],
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
    archived_seeds=_G3_SEEDS,
    archived_by_seed={42: 21.8833, 123: 21.7828, 456: 22.3740},
    archived_mean=22.0134, archived_cost=_G3_COST,
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

# ------------------------------------- the decoder floor, pre-registered
#: Bands whose per-seed tolerance was DECLARED rather than measured, and which
#: may therefore be explained after the fact by a floor that is measured.
#: Only the decoder qualifies: the vision bands already carry a measured
#: same-seed floor (0.040-0.170pp over four dense-control pairs) inside their
#: own widening, so explaining them by one again would be counting it twice.
BANDS_WITH_A_DECLARED_TOLERANCE = ("G3-paper",)

#: Pre-registered 2026-11-20, BEFORE the re-run of R3 and before job 23898426
#: that measures the floor landed. The rule, in full:
#:
#: If a band in `BANDS_WITH_A_DECLARED_TOLERANCE` fails, every reason it
#: failed is a per-seed bound, and every seed that is out of its bounds sits
#: within the floor measured by re-running the archived script at a fixed
#: seed of ITS OWN archived value, then the band is recorded as
#: **FAILED, EXPLAINED BY MEASURED NOISE**, with both numbers, and it does not
#: block the tag. If any deviation exceeds that floor, or the mean or the cost
#: is out of band, the rule does not apply and the failure stands.
#:
#: What this rule will not do: it will not rescue a band whose MEAN is out of
#: band, because a mean that moves is not single-seed noise; it will not apply
#: without a floor measured on this exact cell; and it never widens the band
#: itself, so the recorded verdict stays FAILED.
EXPLAINED_BY_NOISE = "FAILED, EXPLAINED BY MEASURED NOISE"


def measured_floor(rerun_ppl: float, archived_ppl: float) -> float:
    """``|rerun - archive|`` for one seed of one cell, the same-seed floor."""
    return abs(float(rerun_ppl) - float(archived_ppl))


def explain_by_noise(verdict: dict, band, floor: Optional[float],
                     floor_source: str = "") -> dict:
    """Apply the pre-registered rule to one failed band's verdict.

    Returns the verdict with ``status``, ``explained`` and ``blocking`` set.
    A band that passed, a band without a declared tolerance, or a missing
    floor all leave the verdict exactly as it was, which is the conservative
    direction in every case.
    """
    out = dict(verdict)
    out.setdefault("status", "PASS" if verdict.get("passed") else "FAIL")
    out.setdefault("explained", False)
    out["blocking"] = not verdict.get("passed")
    if verdict.get("passed") or band.name not in BANDS_WITH_A_DECLARED_TOLERANCE:
        return out
    if floor is None:
        out["why"] = list(out.get("why", [])) + [
            "no measured same-seed floor is available for this cell, so the "
            "declared tolerance stands and the failure stands with it"]
        return out
    reasons = list(verdict.get("why", []))
    if not reasons or any(not r.startswith("seed ") for r in reasons):
        out["why"] = reasons + [
            "not explainable by single-seed noise: %s"
            % "; ".join(r for r in reasons if not r.startswith("seed "))]
        return out
    # Matched BY SEED, never by position: the cells arrive in file-system
    # order, so zipping would compare one seed's result against another
    # seed's archive and call the difference noise.
    archived = dict(band.archived_by_seed or {})
    per_seed = list(verdict.get("per_seed") or ())
    seeds = [int(x) for x in (verdict.get("seeds") or ())]
    if not archived or len(seeds) != len(per_seed):
        out["why"] = reasons + [
            "this cell's per-seed values cannot be matched to the archive by "
            "seed (%d values, %d seed labels, %d archived), so no per-seed "
            "comparison is defined"
            % (len(per_seed), len(seeds), len(archived))]
        return out
    missing = [q for q in seeds if q not in archived]
    if missing:
        out["why"] = reasons + [
            "the archive has no value for seed(s) %s"
            % ", ".join(str(q) for q in missing)]
        return out
    deviations = {q: measured_floor(v, archived[q])
                  for v, q in zip(per_seed, seeds)}
    # The seeds that actually put the band out of its bounds, read from the
    # bounds themselves rather than parsed out of the failure text. Only
    # these have to be inside the floor: a seed that landed in band needs no
    # explaining, and demanding it would block the tag on a seed nobody
    # objected to.
    offending = [q for v, q in zip(per_seed, seeds)
                 if (band.seed_lo is not None and v < band.seed_lo)
                 or (band.seed_hi is not None and v > band.seed_hi)]
    if not offending:
        out["why"] = reasons + [
            "no single seed is out of its bounds, so this failure is not "
            "single-seed noise"]
        return out
    worst = max(deviations[q] for q in offending)
    if worst <= floor:
        out["status"] = EXPLAINED_BY_NOISE
        out["explained"] = True
        out["blocking"] = False
        out["why"] = reasons + [
            "the out-of-band seed(s) %s deviate from their OWN archived seed "
            "by %s, within the MEASURED same-seed floor %.4f%s. Every seed's "
            "deviation: %s. So this failure is noise at the size the archive "
            "itself re-runs at. The band is not widened and the verdict "
            "stays FAILED"
            % (", ".join(str(q) for q in offending),
               ", ".join("%.4f" % deviations[q] for q in offending),
               floor, (" from " + floor_source) if floor_source else "",
               ", ".join("seed %s %.4f" % (q, deviations[q])
                         for q in seeds))]
        return out
    out["why"] = reasons + [
        "the worst out-of-band seed deviates by %.4f, which EXCEEDS the "
        "measured same-seed floor %.4f%s, so this is not noise. Every "
        "seed's deviation: %s"
        % (worst, floor, (" from " + floor_source) if floor_source else "",
           ", ".join("seed %s %.4f" % (q, deviations[q]) for q in seeds))]
    return out


BANDS = {b.name: b for b in (G1_PAPER, G1_DEFAULT, G2_PAPER, G2_DEFAULT,
                             G3_PAPER, G3_DEFAULT)}

#: Which run set each band is scored from. Four sets, six bands.
RUN_SETS = {
    "A": ("G1-paper", "G1-default"),
    "B": ("G2-paper", "G2-default"),
    "C": ("G3-paper",),
    "D": ("G3-default",),
}

#: What a result file may call itself. Only ``"arm"`` is scoreable: a
#: ``"gate"`` cell is the cheap end-to-end check that every stage runs (2000
#: images, 64 optimizer steps) and a ``"micro"`` cell is CPU plumbing on a
#: synthetic model. None of the three can be told apart by its keys, and the
#: one time a gate cell reached the scorer alongside its arms it was scored as
#: a fourth seed of G1 and moved the band's mean by 6.4pp. So the runner
#: writes which one it ran and the scorer refuses a cell that does not say.
CELL_ROLES = ("arm", "gate", "micro")

#: The only role a band may be scored on.
SCOREABLE_ROLE = "arm"

#: The key a result file carries (the ViT strategy, or the decoder mode) to
#: the bands that key is scored against. One place, so the runner and the
#: off-cluster scorer cannot disagree about which band a cell belongs to.
BANDS_BY_CELL = {
    "random": ("G1-paper", "G1-default"),
    "magnitude": ("G2-paper", "G2-default"),
    "paper": ("G3-paper",),
    "default": ("G3-default",),
}


# ----------------------------------------------------------------- provenance
#: The sidecar that says which code a staged package was built from. It sits
#: BESIDE the package, never inside it, so recording it does not change the
#: hash of the thing it describes.
PROVENANCE_NAME = "PROVENANCE.json"


def package_manifest(package_dir) -> dict:
    """SHA-256 per file and one aggregate over ``*.py``, in name order."""
    import hashlib
    from pathlib import Path

    d = Path(package_dir)
    files = sorted(d.glob("*.py"))
    agg = hashlib.sha256()
    per_file = {}
    for f in files:
        digest = hashlib.sha256(f.read_bytes())
        per_file[f.name] = digest.hexdigest()
        agg.update(f.name.encode() + b"\0" + digest.digest())
    return {"n_files": len(files), "aggregate_sha256": agg.hexdigest(),
            "files": per_file}


def package_provenance(module=None) -> dict:
    """What the running job can say about the code it actually imported.

    Derived from the imported module's own location, not from a path handed
    in: the question a gate result has to answer is "which code did THIS run
    use", and the only authority on that is the module object. The recorded
    sidecar is compared against a manifest computed here and now, so a result
    says both what was claimed and what was there.
    """
    from pathlib import Path

    if module is None:
        import sal as module
    pkg = Path(module.__file__).resolve().parent
    root = pkg.parent
    out = {"package_dir": str(pkg), "version": getattr(module, "__version__",
                                                       None)}
    live = package_manifest(pkg)
    out["computed"] = {k: live[k] for k in ("n_files", "aggregate_sha256")}
    side = root / PROVENANCE_NAME
    if not side.exists():
        out["recorded"] = None
        out["matches"] = False
        out["why"] = ("no %s beside the package, so this result cannot say "
                      "which commit it tested" % PROVENANCE_NAME)
        return out
    rec = json.loads(side.read_text(encoding="utf-8"))
    # Everything the sidecar records about WHAT is in the commit is surfaced,
    # not a fixed subset: a field that exists in the sidecar and silently
    # reads None in the result is worse than no field at all.
    out["recorded"] = {k: v for k, v in rec.items() if k != "files"}
    out["matches"] = (rec.get("aggregate_sha256")
                      == live["aggregate_sha256"]
                      and rec.get("n_files") == live["n_files"])
    out["why"] = ("the package on disk is the one the sidecar records"
                  if out["matches"] else
                  "the package on disk is NOT what the sidecar records: "
                  "recorded %s, computed %s"
                  % (rec.get("aggregate_sha256"), live["aggregate_sha256"]))
    return out


def verify_provenance(prov: Optional[dict], target: str = "HEAD",
                      repo=None) -> dict:
    """Is the code that produced a result an ancestor of what we would push?

    A band may not be called passed on a result produced by code that is not
    in the history being published. That is not bureaucracy: the whole purpose
    of the gate is to say that the SHIPPED pipeline reproduces the archived
    cells, and a result from a commit that never ships says nothing about what
    ships.
    """
    import subprocess
    from pathlib import Path

    if not prov:
        return {"ok": False, "why": "the result carries no provenance at all"}
    if not prov.get("matches"):
        return {"ok": False, "why": prov.get("why", "provenance does not match")}
    commit = (prov.get("recorded") or {}).get("commit")
    if not commit:
        return {"ok": False, "why": "the provenance records no commit"}
    if (prov.get("recorded") or {}).get("dirty"):
        return {"ok": False,
                "why": "the package was staged from a dirty working tree, so "
                       "no commit describes what ran"}
    cwd = str(repo or Path(__file__).resolve().parents[1])
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", commit,
                            target], cwd=cwd, capture_output=True, text=True)
    except OSError as e:                                    # no git here
        return {"ok": False, "why": "cannot check ancestry: %s" % e}
    if r.returncode != 0:
        return {"ok": False, "commit": commit,
                "why": "commit %s is not an ancestor of %s, so this result "
                       "was produced by code that is not in what would be "
                       "pushed" % (commit[:12], target)}
    return {"ok": True, "commit": commit,
            "why": "commit %s is an ancestor of %s" % (commit[:12], target)}


def check(name: str, per_seed, cost=None, seeds=None) -> dict:
    if name not in BANDS:
        raise KeyError("unknown band %r; have %s" % (name, sorted(BANDS)))
    return BANDS[name].check(per_seed, cost, seeds=seeds)


def check_all(results: dict, provenance: Optional[dict] = None,
              target: str = "HEAD", repo=None) -> dict:
    """``{band name: {"per_seed": [...], "cost": x}}`` to one verdict.

    With ``provenance``, a band that meets its numbers is still **not**
    passed unless the code that produced it is an ancestor of ``target``. The
    numbers and the code are two separate claims and both have to hold.
    """
    verdicts = {n: check(n, r.get("per_seed", []), r.get("cost"),
                         seeds=r.get("seeds"))
                for n, r in results.items()}
    prov = verify_provenance(provenance, target=target, repo=repo) \
        if provenance is not None else None
    if prov is not None and not prov["ok"]:
        for v in verdicts.values():
            v["passed"] = False
            v["why"] = list(v["why"]) + ["provenance: " + prov["why"]]
    return {"bands": verdicts,
            "passed": (all(v["passed"] for v in verdicts.values())
                       and (prov is None or prov["ok"])),
            "provenance": prov,
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
