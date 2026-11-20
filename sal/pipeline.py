"""CompressionPipeline — the validated path, in one object.

Compressing a model by hand means wiring `SALConfig` → `HeadMasker` → your
training loop → head selection → slicing → a quantization backend → an
evaluation harness. This object does that wiring and measures every stage, so
the output is a deployment decision rather than a number.

**The stages, in order:** probe → scan → compress → `recover()` → validate →
export. `recover()` is where the accuracy in post-hoc pruning comes from and is
new in v0.7.0: at 50% of heads the one-shot model reads 9.56% and two epochs of
recovery reach 77.642% against a dense control trained for the same two epochs.
Before v0.7.0 this pipeline had no such stage and delivered the first of those
numbers, which is why `validate()` now holds an `accuracy_floor` failure on an
unrecovered model rather than refusing it outright.

*The v0.4.0 recipe this object was built around is superseded.* "Fully fine-tune with SAL,
then quantize" came from one model (GPT-2 Medium / SST-2, 355M, no recovery
pass). The ImageNet-1k campaign -- three seeds, full validation split, a dense
control trained for the same number of epochs -- measured that at a fixed head
set SAL's weights are worth **+0.05pp** after a recovery pass, against the
recovery pass itself being worth up to 68pp. If you can fine-tune after pruning,
that is where the accuracy is.

    from sal import CompressionPipeline

    pipe = CompressionPipeline(model, eval_dataset, metric="accuracy")
    print(pipe.scan().recommendation)
    print(pipe.probe().recommendation)
    pipe.compress(pruning=0.50, quantization="int4", slice_heads=True)
    pipe.recover(train_dataset, epochs=2, domain="same as eval")
    print(pipe.validate().summary)
    pipe.export("compressed_model/")
    pipe.report().save("compression_report.pdf")

Guardrails, in order of how much trouble they save:

* **It refuses LoRA/QLoRA models.** SAL works by letting the model reorganize
  around silenced heads; adapters freeze the weights that would do the
  reorganizing. Measured on GPT-2 Medium, SAL under LoRA lost four of six
  compression variants *and* gave up 3.1 points of clean accuracy: one model,
  one task, five seeds, and no recovery pass, which is the scope of that
  figure. It is still a configuration worth refusing loudly rather than
  serving quietly, because the mechanism it disables is the whole method.
* **It warns below ~100M parameters**, where the benefit is marginal.
* **It measures accuracy and size at every stage**, and can stop on an accuracy
  floor instead of handing back a small, broken model.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SMALL_MODEL_PARAMS = 100_000_000

#: The public data archive every figure in this module is cited to.
ARCHIVE_DOI = "10.5281/zenodo.24391200"

#: Every measured figure in this module's user-facing strings, and the file in
#: the public data archive it comes from. Keyed by the figure exactly as the
#: text prints it, without its sign, so the contract test in
#: ``tests/test_routing_text.py`` is a set membership and not a guess: a figure
#: that reaches a user-facing string without an entry here fails the suite.
#:
#: It is a contract, not a check of the numbers. It cannot tell a right figure
#: from a wrong one. What it stops is a figure arriving with nothing behind it,
#: which is how the two claims this release withdrew got in.
VIT = "data/oss/results/lumi_v060/"
PRO = "data/pro/results/"
SOURCES = {
    # the recovery cells, ViT-B/16 / ImageNet-1k, 3 seeds, 2-epoch control
    "9.56%": VIT + "EXP32_RATIO_AT_2EP.md",
    "78.816%": VIT + "EXP40B_RANDOM.md",
    "2.738pp": VIT + "EXP40B_RANDOM.md",
    "77.642%": VIT + "EXP32_RATIO_AT_2EP.md",
    "3.91pp": VIT + "EXP32_RATIO_AT_2EP.md",
    "81.554%": VIT + "CONTROL_CONVENTION.md",
    # head training as a pre-conditioner, exp24
    "0.05pp": VIT + "EXP24_SAL_RECOVERY.md",
    "0.07pp": VIT + "EXP24_SAL_RECOVERY.md",
    # SAL's set-specificity, exp25 and exp27
    "56.18pp": VIT + "EXP25_ALIGNED.md",
    "8pp": VIT + "EXP27_DROPHEAD.md",
    "3.93pp": VIT + "EXP25_ALIGNED.md",
    # the own-set cell at 631M, exp28b
    "0.67pp": VIT + "EXP28B_JEPA_CORRECTED.md",
    # BERT-base, four GLUE tasks, exp34
    "7.03pp": VIT + "EXP34_BERT_GLUE.md",
    "12.58pp": VIT + "EXP34_BERT_GLUE.md",
    "5.15pp": VIT + "EXP34_BERT_GLUE.md",
    "21.68pp": VIT + "EXP34_BERT_GLUE.md",
    # the routing band itself, exp17
    "0.08pp": VIT + "PHASE2H.md",
    "0.11pp": VIT + "PHASE2H.md",
    # FFN neurons at 2 epochs, 7 seeds
    "0.869pp": PRO + "w3_ffn25_magnitude_s*/exp35_mlp_matched.json",
    "1.397pp": PRO + "w3_ffn25_random_s*/exp35_mlp_matched.json",
    "0.528pp": PRO + "EXP35_FFN_AT_2EP.md",
    "1.91pp": PRO + "EXP35_FFN_AT_2EP.md",
    # compile, measured per batch size
    "1.12x": PRO + "EXP16_COMPILE.md",
    "1.50x": PRO + "EXP16_COMPILE.md",
    "2.2x": PRO + "EXP16_COMPILE.md",
    # the sizes the two structures remove, and DropHead's throughput cost
    "16.4%": PRO + "EXP35_FFN_AT_2EP.md",
    "16.5%": VIT + "EXP25_ALIGNED.md",
    "30%": VIT + "EXP34_BERT_GLUE.md",
    "39%": VIT + "EXP34_BERT_GLUE.md",
}

#: Not measurements: the two routing thresholds, and the pruning ratios the
#: caller asked for rather than anything measured at them.
NOT_MEASUREMENTS = ("30pp", "10pp", "25%", "33%", "50%")

# Routing thresholds for probe(), in points of accuracy lost when half the heads
# are removed from the untrained model. Drawn from exp17's routing sweep, whose
# table shows six rows of which only four were measured under its own protocol:
# the ViT-B/16 and I-JEPA rows are imported ImageNet-100 cells, and that page
# says in its own words not to build a claim on a comparison between an exp17
# row and an imported one. So the bands are a heuristic on four comparable
# points, not a calibrated boundary, and the band a model lands in is a reason
# to measure rather than a prediction. What the probe is for is the question
# "how much does this model lose one-shot", which it answers directly.
FRAGILE_DROP_PP = 30.0
ROBUST_DROP_PP = 10.0


class PipelineError(Exception):
    """Raised when the pipeline is asked to do something unsafe or out of order."""


@contextmanager
def _without_hooks(model):
    """Temporarily strip every hook dict on every submodule.

    ``torch.save`` pickles the module graph, and transformers installs
    output-capturing hooks that are local closures — unpicklable, so saving a
    real HF model fails outright. Stripping them is also the right thing on the
    merits: an exported compressed model is supposed to be standalone, and a
    pickled hook is exactly the kind of hidden dependency that promise rules out.
    Restored afterwards so the live model is unchanged.
    """
    saved = []
    for mod in model.modules():
        for key, value in list(mod.__dict__.items()):
            if key.endswith("_hooks") and hasattr(value, "clear"):
                saved.append((mod, key, value))
                mod.__dict__[key] = type(value)()
    try:
        yield
    finally:
        for mod, key, value in saved:
            mod.__dict__[key] = value


# ------------------------------------------------------------------- detection
def detect_adapters(model) -> Optional[str]:
    """Name of the adapter framework wrapping ``model``, or None.

    Detected structurally rather than by importing peft, so this works whether or
    not peft is installed in the caller's environment.
    """
    if hasattr(model, "peft_config") or type(model).__name__.startswith("Peft"):
        return "peft"
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_A") or hasattr(mod, "lora_B"):
            return "peft/LoRA"
        if "lora" in type(mod).__name__.lower():
            return "LoRA"
    return None


# ---------------------------------------------------------------- result types
@dataclass
class ScanSummary:
    """What the structural scanners say before anything is touched."""
    fi_score: float
    absorption_map: dict
    num_layers: int
    num_heads: int
    total_params: int
    quantization: dict
    adapters: Optional[str] = None

    @property
    def recommendation(self) -> str:
        lines = [
            f"{self.num_layers} layers x {self.num_heads} heads, "
            f"{self.total_params / 1e6:.1f}M parameters",
            f"fragility index: {self.fi_score:.3f} (diagnostic only -- FI does "
            f"not predict how much pruning costs; run probe() for that)",
        ]
        elastic = [l for l, c in self.absorption_map.items() if c == "ELASTIC"]
        hub = [l for l, c in self.absorption_map.items() if c == "HUB"]
        lines.append(f"absorption: {len(elastic)} elastic layer(s), {len(hub)} hub layer(s)")
        q = self.quantization
        lines.append(f"size: {q['original_size_mb']:.0f}MB -> "
                     f"{q['int8_size_mb']:.0f}MB at int8, {q['int4_size_mb']:.0f}MB at int4")
        if self.adapters:
            lines.append(f"WARNING: {self.adapters} adapters detected — SAL needs full "
                         "fine-tuning and will be refused.")
        elif self.total_params < SMALL_MODEL_PARAMS:
            lines.append(f"note: under {SMALL_MODEL_PARAMS / 1e6:.0f}M parameters, "
                         "SAL's benefit is marginal.")
        else:
            lines.append("next: probe() -- it measures what pruning actually costs "
                         "this model and routes to the right method. Nothing in this "
                         "scan can tell you that.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"fi_score": round(self.fi_score, 6),
                "absorption_map": {str(k): v for k, v in self.absorption_map.items()},
                "num_layers": self.num_layers, "num_heads": self.num_heads,
                "total_params": self.total_params, "quantization": self.quantization,
                "adapters": self.adapters, "recommendation": self.recommendation}


@dataclass
class ProbeSummary:
    """How far the model falls when half its heads are removed, and what to do.

    This is the router. It replaces the fragility index in that role, because a
    six-architecture sweep showed FI does not predict the regime: BERT-base has a
    *higher* FI than ViT-B/16 and loses 5pp where ViT-B/16 loses 62, while
    DINOv2-large has an FI 25x lower than DeiT-small and collapses to chance.
    Parameter count does not order it either. What does order it is this
    measurement — the model's own loss under the pruning you intend.

    FI remains in the product for diagnosis and certification. It is not a
    router.
    """
    clean: float
    pruned: float
    fraction: float
    heads_removed: int
    route: str
    metric: str = "accuracy"

    @property
    def drop_pp(self) -> float:
        """Points lost, positive meaning worse. Sign-corrected for ``loss``."""
        delta = (self.clean - self.pruned) * 100
        return delta if self.metric == "accuracy" else -delta

    @property
    def recommendation(self) -> str:
        lines = [
            f"probe: {self.heads_removed} heads removed "
            f"({self.fraction:.0%}), {self.metric} "
            f"{self.clean:.4f} -> {self.pruned:.4f}, "
            f"{self.drop_pp:.2f}pp lost",
        ]
        if self.route == "sal_aligned":
            lines += [
                f"FRAGILE (drop > {FRAGILE_DROP_PP:.0f}pp). One question "
                "decides the method, and it is not the band:",
                "  can you fine-tune AFTER pruning?",
                "    yes -> prune at random, then recover. This is the "
                "shipped path:",
                "      compress(pruning=0.5, strategy='random', "
                "slice_heads=True)",
                "      recover(train_dataset, epochs=2, domain=...)",
                "    no  -> train with head masking first, with "
                "mask_mode='drophead':",
                "      SALConfig.auto(model, prune_fraction=0.5, "
                "mask_mode='drophead')",
                "      compress(pruning=0.5, slice_heads=True)",
                "measured on ViT-B/16 / ImageNet-1k, 3 seeds, full 50k val "
                "split, against a dense control trained for the same 2 epochs "
                "(81.554%): at 50% of heads the one-shot model reads 9.56% "
                "and two epochs of recovery reach 78.816% with random "
                "selection (-2.738pp) or 77.642% with magnitude (-3.91pp). "
                "Head training is worth +0.05pp (SAL) or +0.07pp (DropHead) "
                "as a starting point for that recovery, i.e. nothing.",
                "if you cannot recover: DropHead, with no exception. SAL's "
                "adaptation is worth +56.18pp on the one head set it trained "
                "for and 0 to 8pp on any other, and on that same own-set cell "
                "DropHead is +0.67pp above SAL on 3/3 seeds of a 631M "
                "encoder, so the \"SAL when the deployed set is fixed\" "
                "carve-out is withdrawn. SAL's own-set operating point is "
                "-3.93pp against the matched baseline at a 16.5% smaller "
                "model, which is the figure to compare, not its dense row.",
            ]
        elif self.route == "drophead":
            lines += [
                f"INTERMEDIATE ({ROBUST_DROP_PP:.0f}-{FRAGILE_DROP_PP:.0f}pp). "
                "The answer is the same as in the fragile band, because it "
                "does not depend on the band:",
                "  can you fine-tune after pruning? -> prune, then recover(), "
                "and skip head training. On BERT-base across four GLUE tasks "
                "magnitude + 2 epochs of recovery beat DropHead on 8 of 8 "
                "cells, by 0.42 to 7.03pp over 3 seeds.",
                "  if you cannot -> mask_mode='drophead'. It holds no head "
                "set, so the deployed set stays free until deployment, and on "
                "a 631M encoder it is +0.67pp above SAL even on SAL's own "
                "trained-for set, 3/3 seeds.",
                "what DropHead costs, measured on those same GLUE cells: up "
                "to 12.58pp of dense accuracy (MRPC), and 24-30% of training "
                "throughput at 33% of heads rising to 33-39% at 50%.",
            ]
        else:
            lines += [
                f"ROBUST (drop < {ROBUST_DROP_PP:.0f}pp). Head TRAINING buys "
                "nothing measurable here -- but this threshold is the one the "
                "evidence contradicts, so verify rather than trust it:",
                "  compress(pruning=0.5, quantization='int4', slice_heads=True)",
                "  then recover(train_dataset, epochs=2, domain=...): a "
                "post-pruning fine-tune is worth far more than any "
                "training-time method.",
                "measured: SAL and DropHead were indistinguishable on the "
                "models in this band (+0.08pp on BERT-base, +0.11pp on "
                "phi-2), both at exp17's own protocol of 872 evaluation "
                "examples and not the campaign's full split. BUT the same "
                "BERT-base probed at -5.15pp loses -4.17 to -21.68pp one-shot "
                "at 50% of heads across four GLUE tasks, i.e. mostly OUTSIDE "
                "this band. Probe on YOUR task, at the ratio you intend.",
            ]
        lines.append(
            "then: head pruning is the small half of the compression, and "
            "FFN neurons are the cheaper structure. At the same 2 epochs of "
            "recovery and the same matched control, 25% of FFN neurons cost "
            "0.869pp at -16.4% of parameters (7 seeds, range -0.988 to "
            "-0.664) against 2.738pp for 50% of heads at -16.5% (3 seeds), so "
            "heads are about three times more expensive per unit of size "
            "removed. Prune FFN neurons by MAGNITUDE, not at random: random "
            "costs 1.397pp on the same cell (7 seeds, range -1.668 to "
            "-1.212), and the two ranges do not overlap. The inversion worth "
            "knowing is that random is 1.91pp BETTER one-shot and 0.528pp "
            "worse after recovery, so a one-shot benchmark picks the wrong "
            "FFN default. Neuron pruning is not part of sal-torch; see "
            "ROADMAP.md.")
        lines.append("finally: torch.compile. Measured 1.12x on a dense ViT-B/16 at "
                     "batch 32 and 1.50x on a pruned one, and 2.2x at batch 1 where "
                     "pruning alone buys almost nothing.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"clean": self.clean, "pruned": self.pruned,
                "drop_pp": round(self.drop_pp, 4), "fraction": self.fraction,
                "heads_removed": self.heads_removed, "route": self.route,
                "metric": self.metric, "recommendation": self.recommendation}


@dataclass
class Stage:
    name: str
    size_mb: float
    accuracy: Optional[float] = None
    detail: str = ""
    seconds: float = 0.0


@dataclass
class CompressionReport:
    """Every stage, measured — the waterfall from original to deployable."""
    stages: list = field(default_factory=list)
    metric: str = "accuracy"
    model_name: str = ""
    scan: Optional[ScanSummary] = None
    roundtrip_verified: Optional[bool] = None
    recovery: Optional[dict] = None
    control: Optional["RecoveryControl"] = None
    precision: Optional[dict] = None

    @property
    def first(self) -> Optional[Stage]:
        return self.stages[0] if self.stages else None

    @property
    def last(self) -> Optional[Stage]:
        return self.stages[-1] if self.stages else None

    @property
    def size_ratio(self) -> float:
        if not self.stages or not self.last.size_mb:
            return 1.0
        return self.first.size_mb / self.last.size_mb

    @property
    def accuracy_delta(self) -> Optional[float]:
        if self.first is None or self.first.accuracy is None or self.last.accuracy is None:
            return None
        return self.last.accuracy - self.first.accuracy

    # ------------------------------------------------------- matched control
    #: What the report says instead of a cost when it has no control to take
    #: one against. `CONTROL_CONVENTION.md` exists because a delta against the
    #: pretrained checkpoint is the sum of two effects, and a product user
    #: reads one number and ships.
    NO_CONTROL = (
        "the pruning cost is not isolated, because no dense control trained "
        "for the same budget was supplied")
    UNMATCHED = (
        "the pruning cost is not isolated, because the control was trained "
        "for a different budget than the arm")

    @property
    def higher_is_better(self) -> bool:
        return self.metric == "accuracy"

    @property
    def pretrained_accuracy(self) -> Optional[float]:
        """The model as it was handed in. Not a control."""
        return self.first.accuracy if self.first else None

    @property
    def recovered(self) -> bool:
        return any(st.name == "recovered" for st in self.stages)

    @property
    def arm_accuracy(self) -> Optional[float]:
        """The score the cost is taken on: the last measured stage."""
        for st in reversed(self.stages):
            if st.accuracy is not None:
                return st.accuracy
        return None

    @property
    def budget_matched(self) -> Optional[bool]:
        """True only when the control was trained on the arm's own budget."""
        if self.control is None:
            return None
        rec = self.recovery or {}
        return bool(self.control.matches(rec.get("epochs"), rec.get("tokens")))

    @property
    def fine_tuning_gain(self) -> Optional[float]:
        """Control minus pretrained: what the budget bought with nothing pruned.

        In the metric's own units. Absent without a control, because it is
        defined as control minus pretrained and there is no control to
        subtract from.
        """
        if self.control is None or self.pretrained_accuracy is None:
            return None
        return self.control.accuracy - self.pretrained_accuracy

    @property
    def fine_tuning_gain_pp(self) -> Optional[float]:
        g = self.fine_tuning_gain
        return None if g is None or not self.higher_is_better else g * 100

    @property
    def pruning_cost(self) -> Optional[float]:
        """Arm minus control, in the metric's own units.

        **Emitted only when the budgets match.** A 5-epoch arm scored against
        a 2-epoch control was wrong by 0.97 to 1.12pp on three headline cells
        of this project's own campaign, and the mistake survived in committed
        data.

        The sign is the campaign's, not a third convention: the number is
        arm minus control, so for accuracy a **negative** cost means the
        compression cost accuracy (-3.91pp at 50% of heads on ViT-B/16) and
        for perplexity a **positive** cost means it cost perplexity (+0.9939
        for 50% unstructured on OPT-1.3B). :attr:`higher_is_better` says which
        way to read it, and :attr:`cost_statement` spells it out in words.
        """
        if not self.budget_matched or self.arm_accuracy is None:
            return None
        return self.arm_accuracy - self.control.accuracy

    @property
    def pruning_cost_pp(self) -> Optional[float]:
        """The cost in points of accuracy, negative meaning worse.

        ``None`` for a loss or perplexity metric, where points of accuracy are
        not the unit: read :attr:`pruning_cost` there, in which a positive
        number means worse.
        """
        c = self.pruning_cost
        return None if c is None or not self.higher_is_better else c * 100

    @property
    def cost_statement(self) -> str:
        """One line on the cost, or on why there is not one."""
        unit = "pp" if self.higher_is_better else f" {self.metric}"
        if self.arm_accuracy is None:
            return "nothing measured yet"
        level = f"final {self.metric} {self.arm_accuracy:.4f}"
        if self.control is None:
            return f"{level}; {self.NO_CONTROL}"
        if not self.budget_matched:
            budget = self.control.budget
            return (f"{level}; {self.UNMATCHED} (control: {budget[0]}="
                    f"{budget[1]}, arm: epochs="
                    f"{(self.recovery or {}).get('epochs')}, tokens="
                    f"{(self.recovery or {}).get('tokens')})")
        cost = self.pruning_cost
        gain = self.fine_tuning_gain
        shown = (f"pruning cost {cost * 100:+.2f}pp" if self.higher_is_better
                 else f"pruning cost {cost:+.4f} {self.metric}")
        line = (f"{level}; {shown} against a dense control trained for the "
                f"same {self.control.budget[0]} "
                f"({self.control.seeds} seed(s))")
        if gain is not None:
            line += (f"; the budget itself was worth "
                     f"{gain * 100:+.2f}pp with nothing pruned"
                     if self.higher_is_better else
                     f"; the budget itself cost {gain:+.4f} {self.metric} "
                     "with nothing pruned")
        return line

    @property
    def summary(self) -> str:
        if not self.stages:
            return "nothing measured yet"
        f, l = self.first, self.last
        acc = ""
        if f.accuracy is not None and l.accuracy is not None:
            if not f.accuracy:
                acc = f", {l.accuracy:.3f} {self.metric}"
            elif self.recovered:
                # A delta against the model as handed in is now the sum of two
                # effects, the compression and the recovery, and must not read
                # as a cost. The cost line says what is and is not isolated.
                acc = (f", {f.accuracy:.3f} {self.metric} -> "
                       f"{l.accuracy:.3f} after recovery")
            else:
                acc = (f", {f.accuracy:.3f} {self.metric} -> {l.accuracy:.3f} "
                       f"({self.accuracy_delta:+.1%} relative, one-shot)")
        size = (f"{f.size_mb:.0f}MB -> {l.size_mb:.0f}MB "
                f"({self.size_ratio:.1f}x smaller){acc}")
        return f"{size}. {self.cost_statement}" if self.recovered else size

    @property
    def table(self) -> str:
        head = f"{'stage':<22}{'size_mb':>10}{'ratio':>8}{self.metric:>12}{'seconds':>9}"
        lines = [head, "-" * len(head)]
        base = self.first.size_mb if self.first else 0.0
        for s in self.stages:
            ratio = f"{base / s.size_mb:.2f}x" if s.size_mb else "-"
            acc = f"{s.accuracy:.4f}" if s.accuracy is not None else "-"
            lines.append(f"{s.name:<22}{s.size_mb:>10.1f}{ratio:>8}{acc:>12}{s.seconds:>9.1f}")
        lines.append("-" * len(head))
        lines.append(self.summary)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "metric": self.metric,
            "summary": self.summary,
            "size_ratio": round(self.size_ratio, 4),
            "accuracy_delta": (None if self.accuracy_delta is None
                               else round(self.accuracy_delta, 6)),
            "roundtrip_verified": self.roundtrip_verified,
            "stages": [{"name": s.name, "size_mb": round(s.size_mb, 2),
                        "accuracy": (None if s.accuracy is None else round(s.accuracy, 6)),
                        "detail": s.detail, "seconds": round(s.seconds, 2)}
                       for s in self.stages],
            "scan": self.scan.to_dict() if self.scan else None,
            "recovery": self.recovery,
            "precision": self.precision,
            "cost_statement": self.cost_statement,
            "budget_matched": self.budget_matched,
            "control": (None if self.control is None else
                        {"accuracy": self.control.accuracy,
                         "epochs": self.control.epochs,
                         "tokens": self.control.tokens,
                         "seeds": self.control.seeds,
                         "metric": self.control.metric}),
            **({} if self.pruning_cost is None else
               {"pruning_cost": round(self.pruning_cost, 6)}),
            **({} if self.pruning_cost_pp is None else
               {"pruning_cost_pp": round(self.pruning_cost_pp, 4)}),
            **({} if self.fine_tuning_gain is None else
               {"fine_tuning_gain": round(self.fine_tuning_gain, 6)}),
            **({} if self.fine_tuning_gain_pp is None else
               {"fine_tuning_gain_pp": round(self.fine_tuning_gain_pp, 4)}),
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_compression_pdf
            render_compression_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")


# -------------------------------------------------------------------- pipeline
class CompressionPipeline:
    """Diagnose, SAL-train, compress, validate and export — one object."""

    def __init__(self, model, eval_dataset, metric: str = "accuracy", batch_size: int = 8,
                 probe_dataset=None, seed: int = 0, accuracy_floor: Optional[float] = None,
                 model_name: str = "", max_batches: Optional[int] = None,
                 strict_recovery: bool = False):
        if metric not in ("accuracy", "loss"):
            raise ValueError(f"metric must be 'accuracy' or 'loss', got '{metric}'")
        adapters = detect_adapters(model)
        if adapters:
            raise PipelineError(
                f"This model is wrapped in {adapters} adapters. SAL requires full "
                "fine-tuning: the mechanism is the model reorganizing around silenced "
                "heads, and adapters freeze the weights that would do that. Measured on "
                "GPT-2 Medium, SAL under LoRA lost 4 of 6 compression variants and cost "
                "3.1 points of clean accuracy. Either merge the adapters first "
                "(peft: model.merge_and_unload()) and fully fine-tune, or use "
                "RobustnessTest to measure your compression without SAL.")

        self.model = model
        self.eval_dataset = eval_dataset
        self.probe_dataset = probe_dataset if probe_dataset is not None else eval_dataset
        self.metric = metric
        self.batch_size = batch_size
        self.seed = seed
        self.accuracy_floor = accuracy_floor
        self.model_name = model_name
        self.max_batches = max_batches

        # What precision are the numbers about to be measured in? A config
        # that declares a narrower dtype than the parameters is the OPT bug
        # one save-and-reload away, and half-precision weights make a figure
        # incomparable with the campaign's. Warn, never raise: the user handed
        # us this model and may know exactly what it is.
        from sal.precision import check_load_precision
        self.precision = check_load_precision(model, strict=False)

        self._total_params = sum(p.numel() for p in model.parameters())
        if self._total_params < SMALL_MODEL_PARAMS:
            logger.warning(
                f"Model has {self._total_params / 1e6:.1f}M parameters; below "
                f"{SMALL_MODEL_PARAMS / 1e6:.0f}M SAL's benefit is marginal. "
                "Measure with RobustnessTest before committing to it.")

        self._stages: list = []
        self._scan: Optional[ScanSummary] = None
        self._probe: Optional[ProbeSummary] = None
        self._sal_trained = False
        self._compressed = False
        self._pruned_heads: list = []
        self._roundtrip: Optional[bool] = None
        self.strict_recovery = strict_recovery
        self._recovered = False
        self._recovery: Optional[dict] = None
        self._control = None
        self._deferred_floor: Optional[Stage] = None
        self._one_shot_accuracy: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    def _size(self, model=None) -> float:
        from sal.quantize import model_size_mb
        return model_size_mb(self.model if model is None else model)

    def _accuracy(self, model=None) -> float:
        from sal.robustness import _evaluate
        return _evaluate(self.model if model is None else model, self.eval_dataset,
                         self.metric, self.batch_size, self.max_batches)

    def _record(self, name: str, detail: str = "", seconds: float = 0.0,
                measure: bool = True) -> Stage:
        stage = Stage(name=name, size_mb=self._size(),
                      accuracy=self._accuracy() if measure else None,
                      detail=detail, seconds=seconds)
        self._stages.append(stage)
        self._check_floor(stage)
        return stage

    def _below_floor(self, stage: Stage) -> bool:
        if self.accuracy_floor is None or stage.accuracy is None:
            return False
        return (stage.accuracy < self.accuracy_floor if self.metric == "accuracy"
                else stage.accuracy > self.accuracy_floor)

    def _check_floor(self, stage: Stage):
        """Raise below the floor, unless the model is still repairable.

        A compressed model that has not been recovered is **deferred**, not
        passed. Raising here is what made the floor misdirect: at 50% of heads
        the one-shot model reads 9.56% and two epochs of recovery reach 77.6%,
        so a user refused at compress() and told to lower the ratio was being
        refused for damage that was about to be repaired. The check is held and
        applied by ``validate()`` if ``recover()`` never runs, so nothing is
        exported on the strength of a number nobody looked at.
        """
        if not self._below_floor(stage):
            return
        if self._compressed and not self._recovered and stage.name != "original":
            self._deferred_floor = stage
            logger.warning(
                "%s scored %.4f, past the accuracy_floor of %.4f. Held, not "
                "raised: this model has not been repaired yet. Call recover() "
                "before validate(); on the measured configuration two epochs "
                "are worth up to 68pp at 50%% head pruning.",
                stage.name, stage.accuracy, self.accuracy_floor)
            return
        raise PipelineError(self._floor_message(stage))

    def _floor_message(self, stage: Stage) -> str:
        head = (f"Stage '{stage.name}' scored {stage.accuracy:.4f}, past the "
                f"accuracy_floor of {self.accuracy_floor:.4f}. Stopping rather "
                "than returning a model that is small and broken.")
        if self._recovered or stage.name == "original":
            # Recovery has run, so the damage really is the ratio.
            return head + (" This model has been recovered, so the remaining "
                           "damage is the compression itself: try a smaller "
                           "prune_ratio.")
        return head + (
            " Nothing exported. This model has not been repaired: call "
            "recover() before validate() -- two epochs is worth up to 68pp at "
            "50% head pruning on the measured configuration (ViT-B/16 / "
            "ImageNet-1k, 3 seeds, against a dense control trained for the "
            "same 2 epochs). If you intended the one-shot number, lower "
            "accuracy_floor.")

    def _baseline_stage(self):
        if not self._stages:
            self._record("original",
                         detail=f"{self._total_params / 1e6:.1f}M params, "
                                f"{self.precision['param_dtype']} weights")

    # -------------------------------------------------------------------- probe
    def probe(self, fraction: float = 0.5) -> ProbeSummary:
        """Remove ``fraction`` of the heads at random and measure the damage.

        **Run this first.** It costs two evaluation sweeps and no training, and it
        decides which method is worth paying for — a question that cannot be
        answered from the architecture, the parameter count or the fragility
        index, all three of which were measured and found not to predict it.

        Nothing is changed: the heads are masked inside a context manager and the
        model comes back as it was. The draw is uniform per layer, matching what
        ``slice_heads()`` can express, so the probe measures the perturbation you
        would actually deploy.

        Read ``ProbeSummary.recommendation`` for the route, or ``.route`` for one
        of ``"sal_aligned"`` / ``"drophead"`` / ``"skip_heads"``.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0,1), got {fraction}")
        from sal import SALConfig
        from sal.masker import HeadMasker

        # Captured before anything evaluates: the shared evaluation helper leaves
        # the model in eval mode, and a probe is a read-only measurement that
        # should hand the model back exactly as it was found.
        was_training = self.model.training
        self._baseline_stage()
        clean = self._stages[0].accuracy
        if clean is None:                       # metric was never measured
            clean = self._accuracy()

        cfg = SALConfig.auto(self.model, prune_fraction=fraction,
                             uniform_per_layer=True)
        masker = HeadMasker(self.model, cfg, seed=self.seed)
        self.model.eval()
        masker.install()
        try:
            # activate() prunes the full target in one go, which is what a probe
            # wants: no schedule, no accumulation, just the end state.
            masker.activate()
            pruned = self._accuracy()
            removed = len(masker.pruned_heads())
        finally:
            masker.remove()
            self.model.train(was_training)

        delta = (clean - pruned) * 100
        drop = delta if self.metric == "accuracy" else -delta
        route = ("sal_aligned" if drop > FRAGILE_DROP_PP
                 else "skip_heads" if drop < ROBUST_DROP_PP
                 else "drophead")
        self._probe = ProbeSummary(clean=clean, pruned=pruned, fraction=fraction,
                                   heads_removed=removed, route=route,
                                   metric=self.metric)
        logger.info("probe: %+.2fpp at %.0f%% of heads -> %s",
                    drop, fraction * 100, route)
        return self._probe

    # --------------------------------------------------------------------- scan
    def scan(self) -> ScanSummary:
        """Diagnose the model before touching it. Safe to call at any point."""
        from sal import arch_support
        from sal.fi import _infer_num_heads, compute_fi, extract_activation_graph
        from sal.plasticity import PlasticityScanner
        from sal.quantize import quantize_info

        self._baseline_stage()
        nh = _infer_num_heads(self.model)
        nl = len(arch_support.get_output_projections(self.model))
        adj = extract_activation_graph(self.model, self.probe_dataset,
                                       num_samples=min(200, 10 * self.batch_size),
                                       batch_size=self.batch_size)
        pmap = PlasticityScanner(self.model, self.probe_dataset,
                                 num_samples=min(200, 10 * self.batch_size),
                                 batch_size=self.batch_size).scan()
        self._scan = ScanSummary(
            fi_score=compute_fi(adj), absorption_map=dict(pmap.absorption_map),
            num_layers=nl, num_heads=nh, total_params=self._total_params,
            quantization=quantize_info(self.model), adapters=None)
        return self._scan

    # ---------------------------------------------------------------- sal_train
    def sal_train(self, train_dataset, epochs: int = 3, prune_fraction: float = 0.33,
                  lr: float = 5e-5, batch_size: Optional[int] = None,
                  optimizer=None, max_grad_norm: Optional[float] = 1.0) -> "CompressionPipeline":
        """Fully fine-tune with SAL head masking active.

        Every parameter is trained — that is the point. The masker is removed
        before returning, so the model comes back dense with the adaptation baked
        into the weights.
        """
        from torch.optim import AdamW

        from sal.config import SALConfig
        from sal.fi import _iter_data, _to_dev
        from sal.masker import HeadMasker

        adapters = detect_adapters(self.model)
        if adapters:
            raise PipelineError(f"{adapters} adapters appeared on the model; SAL "
                                "requires full fine-tuning.")

        self._baseline_stage()
        bs = batch_size or self.batch_size
        t0 = time.time()
        torch.manual_seed(self.seed)

        from sal import arch_support
        from sal.fi import _infer_num_heads
        config = SALConfig(num_layers=len(arch_support.get_output_projections(self.model)),
                           num_heads_per_layer=_infer_num_heads(self.model),
                           prune_fraction=prune_fraction)
        device = next(self.model.parameters()).device
        masker = HeadMasker(self.model, config, seed=self.seed)
        masker.install()

        opt = optimizer or AdamW(self.model.parameters(), lr=lr)
        batches = list(_iter_data(train_dataset, bs))
        total_steps = max(1, len(batches) * epochs)
        self.model.train()
        step = 0
        try:
            for _ in range(epochs):
                for batch in batches:
                    masker.step(step, total_steps)
                    batch = _to_dev(batch, device)
                    out = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                    loss = out.loss if hasattr(out, "loss") else out
                    loss.backward()
                    # Silencing a third of the heads makes gradients spikier than
                    # normal fine-tuning; SALTrainer clips by default and this
                    # path should not silently differ from it.
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    step += 1
            stats = masker.stats
        finally:
            masker.remove()
            self.model.eval()

        self._sal_trained = True
        self._record("sal_trained",
                     detail=f"{stats['pruned_heads']}/{stats['total_heads']} heads silenced, "
                            f"{epochs} epochs, full fine-tune",
                     seconds=time.time() - t0)
        return self

    # ----------------------------------------------------------------- compress
    SELECTION_STRATEGIES = ("magnitude", "random", "fi_guided")

    def _head_norms(self) -> dict:
        """L2 norm of each head's slice of its layer's output projection."""
        from sal import arch_support
        from sal.fi import _infer_num_heads

        nh = _infer_num_heads(self.model)
        out: dict = {}
        for li, proj in enumerate(arch_support.get_output_projections(self.model)):
            w = proj.weight.detach().abs().float()
            is_conv1d = type(proj).__name__ == "Conv1D"
            width = w.shape[0] if is_conv1d else w.shape[1]
            hd = width // nh
            for h in range(nh):
                sl = slice(h * hd, (h + 1) * hd)
                block = w[sl, :] if is_conv1d else w[:, sl]
                out[(li, h)] = float(block.norm())
        return out

    def select_heads(self, fraction: float, strategy: str = "random") -> list:
        """Which heads to remove, at a matched total count across strategies.

        ``random`` (default)
            A random set within each layer, seeded. **The default because it is
            measured to be better after recovery, not because it is cheap**: at
            50% of heads on ViT-B/16 / ImageNet-1k, random costs 2.738pp
            against the matched 2-epoch control where magnitude costs 3.91pp,
            and the two seed ranges do not overlap on 3/3 seeds. At 25% and 33%
            the ranges DO overlap (0.101pp and 0.226pp apart), so there the
            difference is not a result and either will do. It is also the
            removal distribution head-masking methods train against.
        ``magnitude``
            Lowest output-projection slice norm within each layer. The obvious
            heuristic, and the worse one on heads by 1.174pp at 50%. Note that
            the answer is the opposite for FFN neurons, where magnitude wins by
            0.528pp over 7 seeds with disjoint ranges: one criterion does not
            cover both structures, and this package stopped assuming it would.
        ``fi_guided``
            Spend the budget on IMMUNE layers first, then BUFFER, never CRITICAL,
            ranking by magnitude within a layer. Deliberately **not** uniform —
            concentrating removal where the fragility scan says it is cheap is
            the whole idea — so it cannot be sliced, only masked.
        """
        if strategy not in self.SELECTION_STRATEGIES:
            raise PipelineError(f"strategy must be one of {self.SELECTION_STRATEGIES}, "
                                f"got '{strategy}'")
        from sal import arch_support
        from sal.fi import _infer_num_heads

        nh = _infer_num_heads(self.model)
        nl = len(arch_support.get_output_projections(self.model))
        k = int(round(fraction * nh))
        if k < 1:
            raise PipelineError(f"pruning={fraction} removes no heads from a "
                                f"{nh}-head layer; use a larger fraction.")
        if k >= nh:
            raise PipelineError(f"pruning={fraction} would remove every head.")
        budget = k * nl                      # matched across all three strategies

        if strategy == "magnitude":
            norms = self._head_norms()
            chosen = []
            for li in range(nl):
                ranked = sorted((norms[(li, h)], h) for h in range(nh))
                chosen.extend((li, h) for _, h in ranked[:k])
            return chosen

        if strategy == "random":
            import random as _random
            rng = _random.Random(self.seed)
            chosen = []
            for li in range(nl):
                chosen.extend((li, h) for h in rng.sample(range(nh), k))
            return chosen

        # fi_guided
        from sal.fi import classify_layers, extract_activation_graph

        adj = extract_activation_graph(self.model, self.probe_dataset,
                                       num_samples=min(200, 10 * self.batch_size),
                                       batch_size=self.batch_size)
        layer_map = classify_layers(self.model, adj, num_heads_per_layer=nh)
        rank = {"IMMUNE": 0, "BUFFER": 1, "CRITICAL": 2}
        norms = self._head_norms()

        def cls_of(li) -> str:
            c = layer_map.get(li)
            return getattr(c, "value", str(c))

        if all(cls_of(li) == "CRITICAL" for li in range(nl)):
            raise PipelineError(
                "fi_guided found no IMMUNE or BUFFER layers — every layer is "
                "CRITICAL, so there is nowhere cheap to prune. Use "
                "strategy='magnitude' or lower the pruning fraction.")

        # Spend the budget in fragility order: immune layers first, then buffer,
        # and only spill into critical layers if the budget cannot be met without
        # them. Capped at half a layer, because a "guided" strategy that guts one
        # layer to spare another is not guidance, it is a strawman — and the spill
        # is reported so an unmatched-looking win cannot hide in the total.
        cap = max(1, nh // 2)
        order = sorted(range(nl), key=lambda li: (rank.get(cls_of(li), 1), li))
        chosen: list = []
        spill = 0
        for li in order:
            if len(chosen) >= budget:
                break
            take = min(cap, budget - len(chosen))
            ranked = sorted((norms[(li, h)], h) for h in range(nh))
            picked = [(li, h) for _, h in ranked[:take]]
            chosen.extend(picked)
            if cls_of(li) == "CRITICAL":
                spill += len(picked)
        self._fi_guided_spill = spill
        self._fi_guided_classes = {li: cls_of(li) for li in range(nl)}
        if spill:
            logger.warning(
                f"fi_guided placed {spill} of {len(chosen)} heads in CRITICAL layers: "
                f"IMMUNE/BUFFER layers could not absorb a {fraction:.0%} budget at a "
                f"cap of {cap} heads per layer. Budget is matched to the other "
                "strategies, but the 'never CRITICAL' property does not hold here.")
        if len(chosen) < budget:
            logger.warning(f"fi_guided placed only {len(chosen)} of {budget} heads; "
                           "the comparison is not at a matched count.")
        return chosen

    def compress(self, pruning: Optional[float] = 0.33, quantization: Optional[str] = "int4",
                 slice_heads: bool = True, backend: str = "auto",
                 strategy: str = "random") -> "CompressionPipeline":
        """Apply head pruning and/or quantization, measuring after each.

        With ``slice_heads=True`` the pruned heads are physically removed, so the
        saving is real; with ``False`` they are masked by hook and the model stays
        the same size — useful only for measuring what pruning costs.
        """
        from sal.slicing import slice_heads as do_slice

        self._baseline_stage()
        if not self._sal_trained:
            logger.warning("compress() called without sal_train(); the model has not been "
                           "made compression-resilient, so these numbers describe plain "
                           "post-hoc compression.")

        # Set before the first measured stage, not after the last: the floor
        # defers on a compressed-and-unrecovered model, and a stage recorded
        # inside compress() is exactly that.
        self._compressed = True
        if pruning:
            t0 = time.time()
            heads = self.select_heads(pruning, strategy=strategy)
            self._pruned_heads = heads
            if slice_heads:
                probe = self._probe_batch()
                self.model = do_slice(self.model, heads, verify_input=probe)
                detail = f"{len(heads)} heads removed from the weights ({strategy})"
                name = "pruned+sliced"
            else:
                # Mask exactly the selected set. Re-deriving a random set here
                # would silently discard the strategy that was asked for.
                from sal.compare import _MaskedHeads
                from sal.fi import _infer_num_heads
                self._mask_ctx = _MaskedHeads(self.model, heads, _infer_num_heads(self.model))
                self._mask_ctx.__enter__()
                detail = (f"{len(heads)} heads masked ({strategy}) — "
                          "not removed, size unchanged")
                name = "pruned (masked)"
            self._record(name, detail=detail, seconds=time.time() - t0)

        if quantization:
            from sal.quantize import quantize
            t0 = time.time()
            self.model = quantize(self.model, method=quantization, backend=backend)
            self._reapply_mask()
            info = getattr(self.model, "_sal_quantization", {})
            self._record(f"quantized ({quantization})",
                         detail=f"backend={info.get('backend', '?')}, "
                                f"{info.get('layers_quantized', '?')} layers",
                         seconds=time.time() - t0)

        self._one_shot_accuracy = self._stages[-1].accuracy
        return self

    def _reapply_mask(self):
        """Re-install head masking on the current model object.

        Quantization hands back a new module graph: the Conv1D rewrite and the
        bitsandbytes swap both *replace* the attention projections, so pre-hooks
        registered on the old modules vanish with them. Whether they survive
        depends on the architecture — GPT-2 loses them, a plain nn.Linear model
        keeps them — which makes this exactly the kind of bug that passes on a
        fixture and silently unmasks a real model. Re-installing is the only
        thing that holds in both cases; without it every selection strategy
        measures the same unmasked model and looks identical.
        """
        if getattr(self, "_mask_ctx", None) is None:
            return
        from sal.compare import _MaskedHeads
        from sal.fi import _infer_num_heads
        try:
            self._mask_ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 — the old modules may already be gone
            pass
        self._mask_ctx = _MaskedHeads(self.model, self._pruned_heads,
                                      _infer_num_heads(self.model))
        self._mask_ctx.__enter__()

    def _probe_batch(self):
        """One batch from the eval set, for slicing verification."""
        from sal.fi import _iter_data, _to_dev
        device = next(self.model.parameters()).device
        for batch in _iter_data(self.eval_dataset, self.batch_size):
            batch = _to_dev(batch, device)
            if isinstance(batch, dict):
                return {k: v for k, v in batch.items() if k != "labels"}
            return batch
        return None

    # ------------------------------------------------------------------ recover
    def recover(self, train_dataset, *, domain: str,
                epochs: Optional[int] = 2, tokens: Optional[int] = None,
                lr: float = 5e-5, batch_size: Optional[int] = None,
                max_grad_norm: Optional[float] = 1.0,
                schedule: str = "warmup_cosine", warmup_ratio: float = 0.05,
                max_lr: Optional[float] = None, pct_start: float = 0.10,
                preserve_sparsity: bool = True, masks: Optional[dict] = None,
                control=None, optimizer=None,
                enforce_masks: bool = True) -> "CompressionPipeline":
        """Repair the compressed model, and measure the repair.

        This is where the accuracy in post-hoc pruning comes from. At 50% of
        attention heads a ViT-B/16 on ImageNet-1k reads 9.56% one-shot and
        77.642% after two epochs against a dense control trained for the same
        two epochs; the pipeline used to deliver the first of those numbers.

        ``epochs=2`` is **measured, not optimal**: the pruned curve peaks at 2
        and declines, with 5 below 2 on 3/3 seeds in FP and INT4, and it is 39%
        of the cost of five. The dense control with nothing pruned peaks at
        **1**, so the two curves do not share a knee and neither knee can be
        assumed to transfer. Re-measure on your data rather than trusting the
        default. For a decoder pass ``tokens=`` instead: epochs over a corpus
        shard is not a unit that transfers between scales.

        ``domain`` is required and has no default. The recovery corpus must
        come from the domain the model is evaluated on, because a mismatched
        one is a domain shift that costs rather than a repair that pays:
        measured on OPT-1.3B, 32M tokens of C4 made the **unpruned** model
        worse on raw WikiText-2 by 0.6398 perplexity, and at 6.7B the same
        corpus cost the dense control more than 50% unstructured pruning cost
        it one-shot, so the recovered arm ended worse than its own one-shot
        model on 3 of 3 seeds. Pass ``domain="generic"`` to proceed knowingly;
        it warns and records the choice.

        ``max_grad_norm`` is on, and the gradient norm and the full loss curve
        are logged every step. The research harness that produced the decoder
        cells clipped nothing under a schedule that was still raising the
        learning rate while the model was at its most damaged, and recovery
        diverged there on the most damaged arm on every seed.

        Sources, public data archive DOI 10.5281/zenodo.24391200:
        ``data/oss/results/lumi_v060/EXP29_RECOVERY_EPOCHS.md``,
        ``EXP32_RATIO_AT_2EP.md``, ``EXP33_SPARSE_VS_STRUCTURED.md``,
        ``data/pro/results/EXP36_OPT_SPARSE.md``.
        """
        from sal.recovery import (GENERIC_DOMAIN, GENERIC_DOMAIN_WARNING,
                                  recover_model)

        if not self._compressed:
            raise PipelineError(
                "recover() repairs what compress() damaged, so it runs after "
                "it. Call compress() first, or drop the stage.")
        if not isinstance(domain, str) or not domain.strip():
            raise PipelineError(
                "recover() needs domain=: a short string naming the domain the "
                "recovery data comes from, which is recorded with the result. "
                "There is no default, because a recovery corpus that does not "
                "match the evaluation domain can cost more than the pruning it "
                "repairs. Pass domain='generic' to proceed knowingly.")
        if domain.strip().lower() == GENERIC_DOMAIN:
            logger.warning(GENERIC_DOMAIN_WARNING)

        self._baseline_stage()
        t0 = time.time()
        rec = recover_model(
            self.model, train_dataset, epochs=epochs, tokens=tokens, lr=lr,
            batch_size=batch_size or self.batch_size,
            max_grad_norm=max_grad_norm, schedule=schedule,
            warmup_ratio=warmup_ratio, max_lr=max_lr, pct_start=pct_start,
            preserve_sparsity=preserve_sparsity, masks=masks,
            enforce_masks=enforce_masks, seed=self.seed, optimizer=optimizer)
        rec["domain"] = domain
        rec["one_shot_accuracy"] = self._one_shot_accuracy
        self._control = control
        self._recovery = rec
        self._recovered = True
        self._deferred_floor = None

        budget = f"{epochs} epoch(s)" if tokens is None else f"{tokens} tokens"
        detail = (f"{budget}, domain={domain}, clip="
                  f"{'off' if not max_grad_norm else max_grad_norm}, "
                  f"{schedule}, {rec['steps']} steps")
        if rec["n_masked_params"]:
            detail += f", sparsity_drift={rec['sparsity_drift']:.6g}"
        stage = self._record("recovered", detail=detail, seconds=time.time() - t0)
        rec["accuracy"] = stage.accuracy
        self._check_domain_shift(rec)
        return self

    def _check_domain_shift(self, rec: dict):
        """Warn when the control moved as far as the damage being repaired.

        A recovery whose own dense control degrades by as much as the arm lost
        one-shot is measuring a domain shift, not a repair, and the honest
        output is the one-shot model. Measured at 6.7B: the corpus cost the
        dense control +1.0748 perplexity while 50% unstructured pruning cost
        +1.1987 one-shot, and the recovered arm ended worse than its own
        one-shot model on 3 of 3 seeds.
        """
        rec["domain_shift_dominates"] = None
        control, first = self._control, self.original_accuracy
        one_shot = rec.get("one_shot_accuracy")
        if control is None or first is None or one_shot is None:
            return
        sign = 1.0 if self.metric == "accuracy" else -1.0
        control_loss = (first - control.accuracy) * sign
        one_shot_damage = (first - one_shot) * sign
        rec["control_movement"] = control_loss
        rec["one_shot_damage"] = one_shot_damage
        dominates = control_loss > 0 and control_loss >= one_shot_damage
        rec["domain_shift_dominates"] = dominates
        if dominates:
            logger.warning(
                "the dense control moved against the evaluation metric by %.4f "
                "while the arm's one-shot damage was %.4f, so this recovery is "
                "measuring a domain shift at least as large as the damage it "
                "repairs. The honest output is the one-shot model. Measured "
                "precedent: at 6.7B the recovered arm ended worse than its own "
                "one-shot model on 3 of 3 seeds.", control_loss, one_shot_damage)

    @property
    def original_accuracy(self):
        """The accuracy of the model as it was handed in, if it was measured."""
        return self._stages[0].accuracy if self._stages else None

    # ----------------------------------------------------------------- validate
    def validate(self) -> CompressionReport:
        """The measured waterfall from original to compressed.

        If a floor was set and the compressed model is below it with no
        recovery run, the floor is applied **here**: it was held at
        ``compress()`` because the model was still repairable, and holding it
        past the point where the user asks for the report would be passing a
        number nobody looked at.
        """
        self._baseline_stage()
        if self._compressed and not self._recovered:
            if self.strict_recovery:
                raise PipelineError(
                    "strict_recovery=True and this compressed model has not "
                    "been recovered, so there is nothing honest to score: a "
                    "one-shot number reads as the model's accuracy and is not. "
                    "Call recover(), or construct the pipeline with "
                    "strict_recovery=False if the one-shot number is what you "
                    "want.")
            if self._deferred_floor is not None:
                raise PipelineError(self._floor_message(self._deferred_floor))
        return CompressionReport(stages=list(self._stages), metric=self.metric,
                                 model_name=self.model_name, scan=self._scan,
                                 roundtrip_verified=self._roundtrip,
                                 recovery=self._recovery,
                                 control=self._control,
                                 precision=self.precision)

    def report(self) -> CompressionReport:
        """Alias for :meth:`validate` — the full report object."""
        return self.validate()

    # ------------------------------------------------------------------- export
    def export(self, path: str, verify: bool = True) -> dict:
        """Write the compressed model so it loads without sal-torch installed.

        HuggingFace models are written with ``save_pretrained``; anything else
        falls back to ``torch.save`` of the module. When ``verify`` is set the
        result is reloaded and compared, and the outcome is recorded on the
        report rather than assumed — some architectures derive attention width
        from ``hidden_size`` and cannot rebuild a sliced model from config.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        result = {"path": str(out), "format": None, "roundtrip_verified": None}

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(out))
            result["format"] = "huggingface"
        else:
            with _without_hooks(self.model):
                torch.save(self.model, out / "model.pt")
            result["format"] = "torch.save"

        if verify:
            result["roundtrip_verified"] = self._verify_export(out, result["format"])
            self._roundtrip = result["roundtrip_verified"]
            if result["roundtrip_verified"] is False and result["format"] == "huggingface":
                # A directory that reloads to a *different* model is worse than no
                # export at all, so write a format that does round-trip alongside it
                # rather than leaving the caller with a warning and a broken artifact.
                with _without_hooks(self.model):
                    torch.save(self.model, out / "model.pt")
                result["format"] = "huggingface+torch.save"
                result["fallback"] = str(out / "model.pt")
                result["roundtrip_verified"] = self._verify_export(out, "torch.save")
                self._roundtrip = result["roundtrip_verified"]
                logger.warning(
                    "save_pretrained did not reload identically: architectures that "
                    "derive attention width from hidden_size (GPT-2, BERT) rebuild a "
                    "sliced model at full width. Wrote model.pt alongside it, which "
                    "loads with torch.load and needs transformers but not sal-torch. "
                    f"Round-trip via that file: {result['roundtrip_verified']}.")
        return result

    def _verify_export(self, out: Path, fmt: str) -> Optional[bool]:
        batch = self._probe_batch()
        if batch is None:
            return None
        try:
            with torch.no_grad():
                mine = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
            mine = mine.logits if hasattr(mine, "logits") else mine

            if fmt == "huggingface":
                # Reload through the model's own class — AutoModelForCausalLM would
                # be wrong for classification, encoder, or vision heads.
                other = type(self.model).from_pretrained(str(out))
            else:
                other = torch.load(out / "model.pt", weights_only=False)
            other.eval()
            with torch.no_grad():
                got = other(**batch) if isinstance(batch, dict) else other(batch)
            got = got.logits if hasattr(got, "logits") else got
            if got.shape != mine.shape:
                return False
            return bool((got.float() - mine.float()).abs().max().item() < 1e-4)
        except Exception as e:  # noqa: BLE001 — a failed reload is the answer, not a crash
            logger.info(f"Export verification failed: {e}")
            return False
