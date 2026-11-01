"""CompressionPipeline — the validated path, in one object.

The v0.4.0 benchmarks pointed at one specific recipe: **fully fine-tune with SAL,
then quantize**. On GPT-2 Medium that produced a model with higher accuracy than
the uncompressed baseline at a quarter of the size. Getting there by hand means
wiring `SALConfig` → `HeadMasker` → your training loop → head selection →
slicing → a quantization backend → an evaluation harness, and getting the
training method wrong silently costs about three points of accuracy.

This makes that path the default one, and measures every stage so the output is
a deployment decision rather than a number:

    from sal import CompressionPipeline

    pipe = CompressionPipeline(model, eval_dataset, metric="accuracy")
    print(pipe.scan().recommendation)
    pipe.sal_train(train_dataset, epochs=3, prune_fraction=0.33)
    pipe.compress(pruning=0.33, quantization="int4", slice_heads=True)
    print(pipe.validate().summary)
    pipe.export("compressed_model/")
    pipe.report().save("compression_report.pdf")

Guardrails, in order of how much trouble they save:

* **It refuses LoRA/QLoRA models.** SAL works by letting the model reorganize
  around silenced heads; adapters freeze the weights that would do the
  reorganizing. Measured on GPT-2 Medium, SAL under LoRA lost four of six
  compression variants *and* gave up 3.1 points of clean accuracy. That is a
  configuration worth refusing loudly rather than serving quietly.
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

# Routing thresholds for probe(), in points of accuracy lost when half the heads
# are removed from the untrained model. Drawn from a six-architecture sweep
# (DeiT-small, ViT-B/16, BERT-base, DINOv2-large, I-JEPA ViT-H/14, phi-2) in
# which SAL's lead over DropHead on an identical head set tracked the probe
# almost monotonically: +31.0pp at a 94pp drop down to +0.1pp at 4pp. The bands
# are a usable heuristic on six points, not a calibrated boundary.
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
                f"FRAGILE (drop > {FRAGILE_DROP_PP:.0f}pp). Train with SAL and prune "
                "the set it trained without:",
                "  sal_train(train_dataset, prune_fraction=0.5)  # uniform_per_layer "
                "is the default",
                "  compress(pruning=0.5, slice_heads=True)       # pass pruned_heads()",
                "measured: SAL led DropHead on the identical head set by +31.0pp "
                "(DINOv2-large, -94pp probe), +12.7pp (DeiT-small, -76pp) and "
                "+2.0pp (ViT-B/16, -62pp).",
            ]
        elif self.route == "drophead":
            lines += [
                f"INTERMEDIATE ({ROBUST_DROP_PP:.0f}-{FRAGILE_DROP_PP:.0f}pp). Either "
                "method works and the choice is about control of the head set:",
                "  you choose the deployed set  -> SAL, and prune exactly "
                "pruned_heads()",
                "  the set is out of your hands -> DropHead, which is robust on any "
                "set of that size",
                "measured on the one model in this band (I-JEPA ViT-H/14, -18.9pp "
                "probe): SAL led by +2.6pp on the identical set.",
            ]
        else:
            lines += [
                f"ROBUST (drop < {ROBUST_DROP_PP:.0f}pp). Skip head training "
                "altogether — it buys nothing here:",
                "  compress(pruning=0.5, quantization='int4', slice_heads=True)",
                "measured: SAL and DropHead were indistinguishable on models in this "
                "band (+0.08pp on BERT-base, +0.11pp on phi-2), and DropHead cost "
                "1.1pp of clean accuracy on I-JEPA for no gain.",
            ]
        lines.append(
            "then: head pruning is the small half of the compression. Pruning FFN "
            "neurons removes twice the parameters for less accuracy — at an "
            "identical -16.5%, 25% of neurons cost 3.1pp against 10-12pp for 50% of "
            "heads. Neuron masking is not part of sal-torch; see ROADMAP.md.")
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

    @property
    def summary(self) -> str:
        if not self.stages:
            return "nothing measured yet"
        f, l = self.first, self.last
        acc = ""
        if f.accuracy is not None and l.accuracy is not None:
            acc = (f", {f.accuracy:.3f} {self.metric} -> {l.accuracy:.3f} "
                   f"({self.accuracy_delta:+.1%} relative)"
                   if f.accuracy else f", {l.accuracy:.3f} {self.metric}")
        return (f"{f.size_mb:.0f}MB -> {l.size_mb:.0f}MB "
                f"({self.size_ratio:.1f}x smaller){acc}")

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
                 model_name: str = "", max_batches: Optional[int] = None):
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

    def _check_floor(self, stage: Stage):
        if self.accuracy_floor is None or stage.accuracy is None:
            return
        below = (stage.accuracy < self.accuracy_floor if self.metric == "accuracy"
                 else stage.accuracy > self.accuracy_floor)
        if below:
            raise PipelineError(
                f"Stage '{stage.name}' scored {stage.accuracy:.4f}, past the "
                f"accuracy_floor of {self.accuracy_floor:.4f}. Stopping rather than "
                "returning a model that is small and broken.")

    def _baseline_stage(self):
        if not self._stages:
            self._record("original", detail=f"{self._total_params / 1e6:.1f}M params")

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
            A random set within each layer, seeded. Measured best of the three on
            GPT-2 Medium — 99.3% accuracy retention against magnitude's 96.3%
            (standard model) and 93.7% (SAL-trained) — and better for *both*
            arms, so this is not a SAL-specific default. It is also the removal
            distribution SAL trains against.
        ``magnitude``
            Lowest output-projection slice norm within each layer. The obvious
            heuristic, and measurably the worse one: a small weight norm turns
            out to be a poor proxy for a head the model can spare.
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

        self._compressed = True
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

    # ----------------------------------------------------------------- validate
    def validate(self) -> CompressionReport:
        """The measured waterfall from original to compressed."""
        self._baseline_stage()
        return CompressionReport(stages=list(self._stages), metric=self.metric,
                                 model_name=self.model_name, scan=self._scan,
                                 roundtrip_verified=self._roundtrip)

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
