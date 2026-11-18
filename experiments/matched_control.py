#!/usr/bin/env python3
"""The matched dense control: load it, verify it, and derive costs from it.

Every wave-2 experiment computes its own pruning costs through this module
instead of quoting a constant or leaving the arithmetic to a write-up. Three
defects from the pre-report audit are what it exists to close:

- **exp31's bug.** A 5-epoch arm was scored against a 2-epoch control, which
  reports the pruning cost plus the gap between two fine-tuning schedules as if
  it were the cost alone. Worth 0.29-0.76pp on three headline cells, and the
  wrong values are still live in committed data
  (``results/lumi_v060/CORRECTIONS.json``).
- **exp33 had no control of its own.** It carried a ``REFERENCE`` dict of
  constants, trained no control, and verified nothing. Its +0.05/-0.22pp costs
  were computed by hand from another job's JSON. Correct, but not reproducible
  through the script.
- **exp33's reference keys did not all state their epoch count**, so a reader
  could not tell which rows were comparable.

So: a control is a *measured object with an epoch count and a seed*, a cost may
only be derived when the epoch counts match, and the loader checks the control
it was handed against what the corpus says before anything is derived from it.

    from matched_control import ControlLadder, cost_table

    ladder = ControlLadder.load()                 # the 2-ep and 5-ep controls
    ctrl = ladder.at(epochs=2, seed=42)           # that seed's own control
    row = ctrl.cost(arm_accuracy=0.77642, arm_epochs=2)
    row["pruning_cost_pp"]                        # -3.91 ... only when matched

Reads through ``results_corrections.load_result``, so the retired
``vs_this_control_pp`` fields cannot leak in.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from results_corrections import REPO_ROOT, load_result  # noqa: E402

__all__ = ["Control", "ControlLadder", "ControlError", "cost_table",
           "PRETRAINED", "CONTROL_SOURCES"]

# The pretrained ViT-B/16 checkpoint on this dataset. Deterministic: one number,
# no seed. It is NOT a control -- see CONTROL_CONVENTION.md -- and is held here
# only so the fine-tuning gain can be reported beside the pruning cost.
PRETRAINED = {"accuracy": 0.7844, "int4_accuracy": 0.77228}

#: The model every figure in CONTROL_SOURCES belongs to. A control is only
#: a control for the model it was measured on, and the ladder refuses to
#: hand a ViT control to an OPT arm.
DEFAULT_MODEL = "google/vit-base-patch16-224"

# The controls whose values are PUBLISHED and therefore verifiable. These
# are checked against PUBLISHED on load; discovery below can find more, but
# finding is not the same as having been checked.
CONTROL_SOURCES: dict[int, list[str]] = {
    1: [f"results/w3_ctrl1_s{s}/exp31_dense_control.json"
        for s in (42, 123, 456)],
    2: [f"results/lumi_v060/exp32_ctrl_s{s}/exp31_dense_control.json"
        for s in (42, 123, 456)],
    3: [f"results/w3_ctrl3_s{s}/exp31_dense_control.json"
        for s in (42, 123, 456)],
    5: ["results/lumi_v060/exp31_s42/exp31_dense_control.json"],
    10: [f"results/w4_ctrl10_s{s}/exp31_dense_control.json"
         for s in (42, 123, 456)],
}

#: Where a dense control may live. Globbed, then each candidate is read and
#: believed only about itself: the model, the budget and the seed come from
#: the file, never from its path, so a run written to an unexpected
#: directory is still found and a mislabelled one is still rejected.
CONTROL_GLOBS = (
    "results/lumi_v060/*/exp31_dense_control.json",
    "results/lumi_v060/*/*dense_control*.json",
    "results/*/exp31_dense_control.json",
    "results/w3_ctrl*/exp31_dense_control.json",
)


def discover_controls(root: Path | None = None, model: str | None = None,
                      globs: Iterable[str] = CONTROL_GLOBS) -> list[dict]:
    """Every dense control on disk, as records keyed by what they are.

    A record is ``{model, epochs, seed, accuracy, int4_accuracy, source}``.
    Only files that actually describe a dense control are returned: one
    with ``heads_removed`` greater than zero, or with no ``control`` block,
    is not a control whatever its filename says.
    """
    root = root or REPO_ROOT
    seen, out = set(), []
    for pat in globs:
        for fp in sorted(root.glob(pat)):
            if fp in seen:
                continue
            seen.add(fp)
            try:
                d = load_result(fp, strict=False)
            except (OSError, ValueError):
                continue
            c = d.get("control")
            if not isinstance(c, dict) or c.get("accuracy") is None:
                continue
            if d.get("heads_removed"):
                # Pruned: not a dense control, whatever it is called.
                continue
            m = d.get("model") or DEFAULT_MODEL
            if model is not None and m != model:
                continue
            ep = c.get("epochs", d.get("epochs"))
            if ep is None:
                continue
            out.append({
                "model": m, "epochs": int(ep), "seed": d.get("seed"),
                "accuracy": c["accuracy"],
                "int4_accuracy": c.get("int4_accuracy"),
                "source": _rel(fp, root),
            })
    return out

# What the write-ups publish, so a drifting control is caught rather than
# silently redefining every cost downstream of it. Tolerance is the measured
# seed spread, not a guess: the three 2-epoch controls span 0.214pp.
PUBLISHED = {
    # The 1-epoch and 3-epoch rungs were measured in wave 3 and sat
    # uncopied on scratch until 2026-11-17; the 10-epoch one landed in wave
    # 4. Registering them is what makes the ladder verifiable at every
    # length rather than only at the two the first write-ups quoted, and it
    # is the rung at ONE epoch, not two, where the dense control peaks.
    1: {"mean": 0.81727, "int4_mean": 0.80567, "seeds": 3,
        "tolerance_pp": 0.25,
        "source": "sal-torch/CLAUDE.md control ladder (81.738 / 81.744 / "
                  "81.700)"},
    2: {"mean": 0.81554, "int4_mean": 0.80273, "seeds": 3,
        "tolerance_pp": 0.25,
        "source": "results/lumi_v060/EXP32_RATIO_AT_2EP.md"},
    3: {"mean": 0.81158, "int4_mean": 0.79821, "seeds": 3,
        "tolerance_pp": 0.25,
        "source": "sal-torch/CLAUDE.md control ladder (81.074 / 81.202 / "
                  "81.198)"},
    5: {"mean": 0.80556, "int4_mean": 0.78956, "seeds": 1,
        "tolerance_pp": 0.25,
        "source": "results/lumi_v060/EXP31 (single seed); the published tables "
                  "use exp25 arm 1a's three-seed 80.44% instead"},
    10: {"mean": 0.79971, "int4_mean": 0.78400, "seeds": 3,
         "tolerance_pp": 0.25,
         "source": "commit 486eddd (80.058 / 79.846 / 80.008)"},
}

#: Every rung the registry carries, in order. DERIVED from CONTROL_SOURCES
#: rather than written out, so a rung added there is in the ladder, in the
#: default load and in the tests at once -- and PUBLISHED cannot come to
#: list a budget the default ladder does not load, which is how adding the
#: 1-, 3- and 10-epoch rungs first broke
#: test_published_registry_matches_the_loaded_corpus.
LADDER_EPOCHS = tuple(sorted(CONTROL_SOURCES))


# On LUMI the jobs write to ``results/<exp>/`` and the repository stores the
# rapatriated copies at ``results/lumi_v060/<exp>/``. A registry entry is
# written in repository form and resolved against both layouts, so the same
# script computes the same costs on the cluster and on a workstation. Anything
# else and a job silently falls back to a different control.
_PREFIX_ALIASES = (("results/lumi_v060/", "results/"),)


def _candidates(rel: str, root: Path) -> list[Path]:
    out = [root / rel]
    for frm, to in _PREFIX_ALIASES:
        if rel.startswith(frm):
            out.append(root / (to + rel[len(frm):]))
    return out


def _resolve_source(rel: str, root: Path) -> Path | None:
    for c in _candidates(rel, root):
        if c.exists():
            return c
    return None


def _rel(path: Path, root: Path) -> str:
    """A repo-relative source string, or the full path when it is outside."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


class ControlError(RuntimeError):
    """A control is missing, mismatched, or has drifted from its published value."""


@dataclass
class Control:
    """One dense model trained for ``epochs`` with nothing pruned."""

    epochs: int
    accuracy: float
    int4_accuracy: float | None
    seed: int | None
    seeds: int = 1
    source: str = ""
    per_seed: dict[int, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- gains
    @property
    def fine_tuning_gain_pp(self) -> float:
        """Control - pretrained: what the epochs bought with nothing pruned."""
        return (self.accuracy - PRETRAINED["accuracy"]) * 100

    @property
    def fine_tuning_gain_int4_pp(self) -> float | None:
        if self.int4_accuracy is None:
            return None
        return (self.int4_accuracy - PRETRAINED["int4_accuracy"]) * 100

    # ---------------------------------------------------------------- costs
    def cost(self, arm_accuracy: float, arm_epochs: int,
             arm_int4_accuracy: float | None = None,
             label: str = "") -> dict:
        """One reportable row for an arm.

        ``pruning_cost_pp`` is present **only** when ``arm_epochs`` equals this
        control's epoch count. When they differ the row carries
        ``epochs_matched: False`` and an explanation instead, because the
        subtraction would mix two effects. ``fine_tuning_gain_pp`` is always
        reported separately, and the arm's delta against the pretrained
        checkpoint is labelled as the **sum** of the two so it cannot be
        misread as a pruning result.
        """
        matched = arm_epochs == self.epochs
        row = {
            "label": label,
            "accuracy": arm_accuracy,
            "arm_epochs": arm_epochs,
            "control_epochs": self.epochs,
            "control_accuracy": self.accuracy,
            "control_seeds": self.seeds,
            "epochs_matched": matched,
            "fine_tuning_gain_pp": self.fine_tuning_gain_pp,
            "vs_pretrained_pp": (arm_accuracy - PRETRAINED["accuracy"]) * 100,
            "vs_pretrained_is": "the SUM of the fine-tuning gain and the "
                                "pruning cost; not a pruning cost",
        }
        if arm_int4_accuracy is not None:
            row["int4_accuracy"] = arm_int4_accuracy
            row["fine_tuning_gain_int4_pp"] = self.fine_tuning_gain_int4_pp
            row["vs_pretrained_int4_pp"] = (
                arm_int4_accuracy - PRETRAINED["int4_accuracy"]) * 100
        if matched:
            row["pruning_cost_pp"] = (arm_accuracy - self.accuracy) * 100
            if arm_int4_accuracy is not None and self.int4_accuracy is not None:
                row["pruning_cost_int4_pp"] = (
                    arm_int4_accuracy - self.int4_accuracy) * 100
        else:
            row["why_no_pruning_cost"] = (
                f"the arm ran {arm_epochs} epochs and this control ran "
                f"{self.epochs}; a {self.epochs}-epoch control is needed. The "
                "dense arm alone moves 1.11pp between 2 and 5 epochs on this "
                "recipe, which is larger than most of the effects measured. "
                "See results/lumi_v060/CONTROL_CONVENTION.md.")
        return row

    def describe(self) -> str:
        n = f"{self.seeds} seed" + ("s" if self.seeds != 1 else "")
        where = f", seed {self.seed}" if self.seed is not None else ""
        return (f"{self.epochs}-epoch dense control: "
                f"{self.accuracy * 100:.3f}% ({n}{where}), "
                f"fine-tuning gain {self.fine_tuning_gain_pp:+.2f}pp")


@dataclass
class ControlLadder:
    """Every control the corpus has, for ONE model, keyed by epoch count."""

    controls: dict[int, Control]
    per_seed: dict[tuple[int, int], Control] = field(default_factory=dict)
    #: A control belongs to the model it was measured on. Part of the key,
    #: not decoration: at() refuses to hand this ladder's control to an arm
    #: of a different model.
    model: str = DEFAULT_MODEL
    #: Second measurements of a control that is already registered. The
    #: registered value is used; these are kept so the run-to-run variance
    #: of the control can be read off rather than guessed at.
    repeats: list = field(default_factory=list)

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, epochs: Iterable[int] | None = None, verify: bool = True,
             root: Path | None = None, model: str | None = None,
             discover: bool = True) -> "ControlLadder":
        """The control ladder for one model.

        ``model`` defaults to ViT-B/16, which is what every published
        figure here is. It is part of the key, not decoration: handing an
        OPT arm a ViT control would be a silent category error, so the
        ladder records which model it is for and ``at()`` refuses a
        mismatch.

        ``discover`` adds every control found on disk for that model, so a
        new seed is visible the moment its job has written its JSON. The
        registered sources are still the ones ``verify()`` checks: being
        found is not being checked.
        """
        root = root or REPO_ROOT
        model = model or DEFAULT_MODEL
        # Every registered rung by default. A fixed (2, 5) meant that
        # registering a rung left the default ladder behind it.
        epochs = LADDER_EPOCHS if epochs is None else epochs
        controls: dict[int, Control] = {}
        per_seed: dict[tuple[int, int], Control] = {}
        found = discover_controls(root, model=model) if discover else []
        for ep in epochs:
            registered = CONTROL_SOURCES.get(ep, [])
            resolved = [(rel, _resolve_source(rel, root)) for rel in registered]
            missing = [rel for rel, got in resolved if got is None]
            if not registered or missing:
                tried = [str(c) for rel in missing
                         for c in _candidates(rel, root)]
                raise ControlError(
                    f"{ep}-epoch control: "
                    + ("no source registered" if not registered else
                       f"missing {missing}; tried {tried}"))
            paths = [got for _, got in resolved]
            accs, int4s, seeds = [], [], []
            for p in paths:
                d = load_result(p)
                if d.get("epochs") != ep:
                    raise ControlError(
                        f"{p} says epochs={d.get('epochs')!r}, expected {ep}. "
                        "The source registry and the data disagree; one of "
                        "them is wrong and no cost may be derived until it is "
                        "resolved.")
                c = d["control"]
                if c.get("epochs") not in (None, ep):
                    raise ControlError(
                        f"{p}: control/epochs is {c['epochs']!r}, not {ep}")
                accs.append(c["accuracy"])
                if c.get("int4_accuracy") is not None:
                    int4s.append(c["int4_accuracy"])
                seed = d.get("seed")
                seeds.append(seed)
                per_seed[(ep, seed)] = Control(
                    epochs=ep, accuracy=c["accuracy"],
                    int4_accuracy=c.get("int4_accuracy"), seed=seed, seeds=1,
                    source=_rel(p, root))
            controls[ep] = Control(
                epochs=ep,
                accuracy=statistics.fmean(accs),
                int4_accuracy=statistics.fmean(int4s) if int4s else None,
                seed=None, seeds=len(accs),
                source="; ".join(CONTROL_SOURCES[ep]),
                per_seed={s: a for s, a in zip(seeds, accs) if s is not None})

        # Discovered controls fill in seeds and budgets the registry does
        # not list. They never REPLACE a registered one: the registered
        # values are the verified ones, and a discovered file that claims
        # the same (epochs, seed) is a duplicate to be reported, not a
        # silent override.
        # A second measurement of the same (model, budget, seed) is a
        # REPEAT, not a contradiction. Wave 3 re-ran the 2-epoch controls
        # for seeds 42/123/456, and the first version of this guard read
        # that as two files disagreeing and refused to load at all.
        #
        # The registered, verified value stays authoritative -- it is the
        # one checked against the published figure -- and the repeat is
        # recorded beside it with its delta, because the run-to-run
        # variance of a control is worth knowing and is otherwise
        # invisible.
        repeats = []
        for rec in found:
            key = (rec["epochs"], rec["seed"])
            if key in per_seed:
                delta = rec["accuracy"] - per_seed[key].accuracy
                if abs(delta) > 1e-12:
                    repeats.append({
                        "epochs": key[0], "seed": key[1],
                        "authoritative": per_seed[key].source,
                        "authoritative_accuracy": per_seed[key].accuracy,
                        "repeat": rec["source"],
                        "repeat_accuracy": rec["accuracy"],
                        "delta_pp": delta * 100,
                    })
                continue
            per_seed[key] = Control(
                epochs=rec["epochs"], accuracy=rec["accuracy"],
                int4_accuracy=rec["int4_accuracy"], seed=rec["seed"],
                seeds=1, source=rec["source"])
        ladder = cls(controls=controls, per_seed=per_seed, model=model,
                     repeats=repeats)
        if verify:
            ladder.verify()
        return ladder

    # ----------------------------------------------------------- verifying
    def verify(self) -> dict:
        """Check each loaded control against what the write-ups publish.

        A control that has drifted silently redefines every cost derived from
        it, so this raises rather than warns.
        """
        report = {}
        for ep, ctrl in self.controls.items():
            pub = PUBLISHED.get(ep)
            if pub is None:
                report[ep] = {"checked": False,
                              "note": "no published value registered"}
                continue
            gap = abs(ctrl.accuracy - pub["mean"]) * 100
            if gap > pub["tolerance_pp"]:
                raise ControlError(
                    f"{ep}-epoch control reads {ctrl.accuracy * 100:.3f}% "
                    f"against the published {pub['mean'] * 100:.3f}% "
                    f"({pub['source']}), a gap of {gap:.3f}pp over a tolerance "
                    f"of {pub['tolerance_pp']}pp. Every cost derived from this "
                    "control would move with it.")
            if ctrl.seeds != pub["seeds"]:
                raise ControlError(
                    f"{ep}-epoch control has {ctrl.seeds} seed(s), published "
                    f"as {pub['seeds']}. The registry and the corpus disagree.")
            report[ep] = {"checked": True, "gap_pp": gap,
                          "seeds": ctrl.seeds, "published": pub["mean"]}
        return report

    # ------------------------------------------------------------- lookups
    def at(self, epochs: int, seed: int | None = None,
           for_model: str | None = None) -> Control:
        """The control for ``epochs``; that seed's own control when given.

        Prefer passing ``seed``. Where an arm is single-seed, scoring it against
        the three-seed mean imports the other two seeds' noise into its cost
        (CONTROL_CONVENTION.md); exp28 measured a 7.56pp spread from evaluation
        shuffling alone.
        """
        if for_model is not None and for_model != self.model:
            raise ControlError(
                "this ladder is for %r and you asked for a control for "
                "%r. A control belongs to the model it was measured on; "
                "scoring an arm of one model against another model's "
                "control is not a cost, it is a category error."
                % (self.model, for_model))
        if seed is not None:
            c = self.per_seed.get((epochs, seed))
            if c is None:
                seeds_here = sorted(s for e, s in self.per_seed if e == epochs
                                    and s is not None)
                epochs_here = sorted(self.controls)
                raise ControlError(
                    f"no {epochs}-epoch control for seed {seed}. "
                    + (f"Seeds at {epochs} epochs: {seeds_here}."
                       if seeds_here else
                       f"There is no {epochs}-epoch control at all; "
                       f"controls exist at {epochs_here} epochs. An arm "
                       f"trained for {epochs} epochs needs a control trained "
                       f"for {epochs} epochs -- see "
                       f"results/lumi_v060/CONTROL_CONVENTION.md."))
            return c
        c = self.controls.get(epochs)
        if c is None:
            raise ControlError(
                f"no {epochs}-epoch control; have {sorted(self.controls)}")
        return c

    def describe(self) -> str:
        return "\n".join(f"  {c.describe()}"
                         for _, c in sorted(self.controls.items()))

    def to_dict(self) -> dict:
        return {
            "pretrained": PRETRAINED,
            "controls": {
                str(ep): {"accuracy": c.accuracy,
                          "int4_accuracy": c.int4_accuracy,
                          "seeds": c.seeds, "per_seed": c.per_seed,
                          "fine_tuning_gain_pp": c.fine_tuning_gain_pp,
                          "fine_tuning_gain_int4_pp": c.fine_tuning_gain_int4_pp,
                          "source": c.source}
                for ep, c in sorted(self.controls.items())},
            "convention": "results/lumi_v060/CONTROL_CONVENTION.md",
            "verified": self.verify(),
        }


def cost_table(rows: list[dict], title: str = "") -> str:
    """Render ``Control.cost`` rows, with unmatched ones visibly set apart."""
    head = (f"{'arm':<26}{'ep':>3}{'top-1':>9}{'cost':>9}"
            f"{'INT4':>9}{'INT4 cost':>11}")
    out = ([title] if title else []) + [head, "-" * len(head)]
    for r in rows:
        cost = (f"{r['pruning_cost_pp']:+8.2f}" if r.get("epochs_matched")
                else "  n/a   ")
        i4 = f"{r['int4_accuracy'] * 100:8.3f}" if "int4_accuracy" in r else "       -"
        i4c = (f"{r['pruning_cost_int4_pp']:+10.2f}"
               if "pruning_cost_int4_pp" in r else "     n/a  ")
        out.append(f"{r.get('label', '')[:26]:<26}{r['arm_epochs']:>3}"
                   f"{r['accuracy'] * 100:>9.3f}{cost}{i4}{i4c}")
    unmatched = [r for r in rows if not r.get("epochs_matched")]
    if unmatched:
        out.append("")
        for r in unmatched:
            out.append(f"  n/a: {r.get('label', '')} -- {r['why_no_pruning_cost']}")
    return "\n".join(out)


if __name__ == "__main__":
    ladder = ControlLadder.load()
    print("control ladder, verified against the write-ups:")
    print(ladder.describe())
    print("\nexample: exp32's P50 arm at 2 epochs, and the same arm misread")
    c2 = ladder.at(2)
    print(cost_table([
        c2.cost(0.77642, arm_epochs=2, arm_int4_accuracy=0.75427,
                label="magnitude P50 + 2 ep"),
        c2.cost(0.77265, arm_epochs=5, label="magnitude P50 + 5 ep"),
    ]))
