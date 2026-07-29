"""Robustness suite — does this model survive real-world compression?

SAL trains models to keep working when attention heads disappear. But most
practitioners do not prune heads: they **quantize**. This module answers the
question that follows from that — how much accuracy does a given model lose
under each compression method people actually deploy, and does a SAL-trained
model lose less?

    from sal import RobustnessTest

    test = RobustnessTest(model, eval_dataset, metric="accuracy")
    report = test.run(methods=["int8", "int4", "head_pruning_33",
                              "neuron_dropout_10"])
    print(report.table)
    print(report.robustness_score)     # aggregate 0-1
    report.save("robustness.pdf")

Methods
-------
``int8``
    Dynamic INT8 quantization of every ``nn.Linear`` via ``torch.ao.quantization``.
    Runs on CPU (that is where dynamic quantization is supported), so the model
    is copied to CPU for this method regardless of where it lives.

``int4``
    4-bit weight-only quantization. Uses bitsandbytes NF4 when it is installed
    and CUDA is available; otherwise falls back to a simulated per-channel
    symmetric INT4 quantize/dequantize, which needs no extra dependency and
    reproduces the accuracy effect of weight-only 4-bit rounding. The backend
    actually used is recorded on the result. Set ``allow_simulated_quant=False``
    to skip the method instead of simulating it.

``head_pruning_<pct>``
    Silences ``<pct>``% of attention heads with the shipped
    :class:`~sal.masker.HeadMasker` — the exact mechanism SAL trains against.

``neuron_dropout_<pct>``
    Zeroes ``<pct>``% of the neurons in every feed-forward expansion projection
    at inference time, simulating dead units / noisy hardware. Repeated over
    ``dropout_trials`` random fault patterns; mean and standard deviation are
    reported.

Survival
--------
A method is *survived* when the relative degradation stays within
``survival_threshold`` (default 5% of the clean baseline score).
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_METHODS = ("int8", "int4", "head_pruning_33", "head_pruning_50",
                   "neuron_dropout_10", "neuron_dropout_20")

# Module-name fragments that mark a feed-forward block. Matched case-insensitively
# against the dotted module name; the expansion projection inside is then found by
# out_features > in_features.
_FFN_HINTS = ("mlp", "ffn", "feed_forward", "feedforward", "intermediate",
              "fc", "lin1", "lin2", "c_fc", "up_proj", "gate_proj", "dense_h_to_4h")

_HEAD_PRUNING_RE = re.compile(r"^head_pruning_(\d+)$")
_NEURON_DROPOUT_RE = re.compile(r"^neuron_dropout_(\d+)$")


# ------------------------------------------------------------------- evaluation
def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # fully quantized / parameterless — assume CPU
        return torch.device("cpu")


def _iter_batches(dataset, batch_size):
    from sal.fi import _iter_data
    return _iter_data(dataset, batch_size)


def _evaluate(model, dataset, metric: str, batch_size: int,
              max_batches: Optional[int] = None) -> float:
    """Clean evaluation loop. Returns accuracy (higher better) or loss (lower better)."""
    from sal.fi import _to_dev
    model.eval()
    device = _device_of(model)
    correct = total = 0
    loss_sum, n_loss = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(_iter_batches(dataset, batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            batch = _to_dev(batch, device)
            out = model(**batch) if isinstance(batch, dict) else model(batch)
            if metric == "accuracy":
                logits = out.logits if hasattr(out, "logits") else out
                labels = batch["labels"]
                preds = logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
                total += int(labels.numel())
            else:
                loss = out.loss if hasattr(out, "loss") else out
                loss_sum += float(loss)
                n_loss += 1
    if metric == "accuracy":
        return correct / max(total, 1)
    return loss_sum / max(n_loss, 1)


# ------------------------------------------------------------------ quantization
def _quantize_int8(model):
    """Dynamic INT8 over every nn.Linear. Returns a CPU copy; the original is untouched."""
    m = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)


def _fake_quantize_int4_(weight: torch.Tensor) -> None:
    """In-place symmetric per-output-channel INT4 quantize/dequantize of a weight."""
    w = weight.data
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / 7.0     # signed int4 range [-8, 7]
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / scale), -8.0, 7.0)
    w.copy_((q * scale).reshape(w.shape))


def _quantize_int4_simulated(model):
    """Weight-only INT4 on every nn.Linear, simulated by quantize/dequantize."""
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            _fake_quantize_int4_(mod.weight)
    return m


def _quantize_int4_bnb(model):
    """Real NF4 quantization via bitsandbytes. Raises if unavailable or unsupported."""
    import bitsandbytes as bnb

    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes 4-bit needs CUDA")
    m = copy.deepcopy(model).eval()

    def swap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                        bias=child.bias is not None,
                                        compute_dtype=torch.float16, quant_type="nf4")
                new.weight = bnb.nn.Params4bit(data=child.weight.data.clone(),
                                               requires_grad=False, quant_type="nf4")
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
                setattr(parent, name, new)
            else:
                swap(child)

    swap(m)
    return m.cuda()


# ------------------------------------------------------------------ head pruning
def _model_head_shape(model):
    from sal import arch_support
    from sal.fi import _infer_num_heads
    nl = len(arch_support.get_output_projections(model))
    nh = _infer_num_heads(model)
    if nl == 0:
        raise ValueError("No attention output projections found — cannot prune heads.")
    return nl, nh


class _PrunedHeads:
    """Context manager silencing a random ``fraction`` of heads via HeadMasker."""

    def __init__(self, model, fraction: float, seed: int):
        from sal.config import SALConfig
        from sal.masker import HeadMasker
        nl, nh = _model_head_shape(model)
        config = SALConfig(num_layers=nl, num_heads_per_layer=nh, prune_fraction=fraction)
        self.masker = HeadMasker(model, config, seed=seed)

    def __enter__(self):
        self.masker.install()
        self.masker.activate()
        return self

    def __exit__(self, *exc):
        self.masker.remove()


# --------------------------------------------------------------- neuron dropout
def _ffn_expansion_linears(model) -> list:
    """(name, module) for every feed-forward expansion projection.

    The expansion projection is the one that widens the hidden state
    (``out_features > in_features``) inside a module whose name looks like a
    feed-forward block. Its output units are the FFN "neurons".
    """
    named = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and m.out_features > m.in_features]
    hinted = [(n, m) for n, m in named if any(h in n.lower() for h in _FFN_HINTS)]
    if hinted:
        return hinted
    # No recognizable FFN naming — fall back to every widening Linear that is not
    # an attention projection.
    from sal import arch_support
    attn_ids = {id(p) for p in arch_support.get_output_projections(model)}
    return [(n, m) for n, m in named if id(m) not in attn_ids]


class _NeuronDropout:
    """Zero a fixed random subset of FFN neurons for the duration of the block.

    The fault pattern is drawn once and held across all batches — dead units,
    not per-batch noise.
    """

    def __init__(self, modules: list, p: float, seed: int):
        self.modules = modules
        gen = torch.Generator().manual_seed(seed)
        self.masks = {}
        for name, mod in modules:
            n = mod.out_features
            k = int(round(p * n))
            mask = torch.ones(n)
            if k > 0:
                mask[torch.randperm(n, generator=gen)[:k]] = 0.0
            self.masks[name] = mask
        self.handles: list = []

    def _hook(self, name):
        def fn(mod, inputs, output):
            mask = self.masks[name].to(output.device, output.dtype)
            return output * mask
        return fn

    def __enter__(self):
        for name, mod in self.modules:
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------- result objects
@dataclass
class MethodRobustness:
    """What one compression method did to one model."""
    method: str
    baseline: float
    after: float
    survived: bool
    time_seconds: float
    std: Optional[float] = None       # across trials, for stochastic methods
    backend: Optional[str] = None     # e.g. "bitsandbytes-nf4", "simulated-int4"
    skipped: bool = False
    note: str = ""
    metric_direction: str = "higher_better"

    @property
    def delta(self) -> float:
        """Signed change in the metric (after - baseline)."""
        return self.after - self.baseline

    @property
    def degradation(self) -> float:
        """Relative loss of quality, positive = worse, regardless of metric direction."""
        return _degradation(self.baseline, self.after, self.metric_direction)

    @property
    def retention(self) -> float:
        """Fraction of clean quality retained, clipped to [0, 1]."""
        return float(np.clip(1.0 - self.degradation, 0.0, 1.0))


def _degradation(baseline: float, after: float, direction: str) -> float:
    # A near-zero baseline has no meaningful relative scale, so fall back to the
    # absolute change rather than dividing by epsilon and reporting nonsense.
    denom = abs(baseline) if abs(baseline) > 1e-9 else 1.0
    raw = (baseline - after) if direction == "higher_better" else (after - baseline)
    return float(raw / denom)


@dataclass
class RobustnessReport:
    """Per-method survival of a single model, plus an aggregate score."""
    results: list
    metric: str
    survival_threshold: float
    model_name: str = ""

    @property
    def evaluated(self) -> list:
        return [r for r in self.results if not r.skipped]

    @property
    def robustness_score(self) -> float:
        """Mean quality retained across every evaluated method, in [0, 1].

        1.0 means no method cost the model anything; 0.0 means every method
        destroyed it. Skipped methods are excluded rather than counted as passes.
        """
        vals = [r.retention for r in self.evaluated]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def survival_rate(self) -> float:
        """Fraction of evaluated methods the model survived."""
        ev = self.evaluated
        return len([r for r in ev if r.survived]) / len(ev) if ev else 0.0

    @property
    def table(self) -> str:
        head = (f"{'method':<18}{'baseline':>10}{'after':>10}{'delta':>10}"
                f"{'std':>8}{'survived':>10}")
        lines = [head, "-" * len(head)]
        for r in self.results:
            if r.skipped:
                lines.append(f"{r.method:<18}{'-':>10}{'-':>10}{'-':>10}{'-':>8}{'skipped':>10}")
                continue
            std = f"{r.std:.3f}" if r.std is not None else "-"
            lines.append(f"{r.method:<18}{r.baseline:>10.4f}{r.after:>10.4f}"
                         f"{r.delta:>+10.4f}{std:>8}{('OK' if r.survived else 'FAIL'):>10}")
        lines.append("-" * len(head))
        lines.append(f"robustness_score={self.robustness_score:.4f}  "
                     f"survived {len([r for r in self.evaluated if r.survived])}"
                     f"/{len(self.evaluated)} methods")
        return "\n".join(lines)

    @property
    def summary(self) -> str:
        name = f"{self.model_name}: " if self.model_name else ""
        return (f"{name}robustness={self.robustness_score:.3f}, "
                f"survived {len([r for r in self.evaluated if r.survived])}/"
                f"{len(self.evaluated)} methods "
                f"(metric={self.metric}, threshold={self.survival_threshold:.0%})")

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "metric": self.metric,
            "survival_threshold": self.survival_threshold,
            "robustness_score": round(self.robustness_score, 6),
            "survival_rate": round(self.survival_rate, 6),
            "results": [
                {"method": r.method, "baseline": round(r.baseline, 6),
                 "after": round(r.after, 6), "delta": round(r.delta, 6),
                 "degradation": round(r.degradation, 6), "retention": round(r.retention, 6),
                 "survived": bool(r.survived), "skipped": bool(r.skipped),
                 "std": None if r.std is None else round(r.std, 6),
                 "backend": r.backend, "note": r.note,
                 "time_seconds": round(r.time_seconds, 3)}
                for r in self.results
            ],
            "summary": self.summary,
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_robustness_pdf
            render_robustness_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")


# --------------------------------------------------------------------- the test
class RobustnessTest:
    """Run a model through a battery of compression methods and score what survives."""

    def __init__(self, model, eval_dataset, metric: str = "accuracy", batch_size: int = 8,
                 survival_threshold: float = 0.05, seed: int = 0, dropout_trials: int = 3,
                 max_batches: Optional[int] = None, allow_simulated_quant: bool = True,
                 model_name: str = ""):
        if metric not in ("accuracy", "loss"):
            raise ValueError(f"metric must be 'accuracy' or 'loss', got '{metric}'")
        if not 0.0 <= survival_threshold <= 1.0:
            raise ValueError(f"survival_threshold must be in [0, 1], got {survival_threshold}")
        self.model = model
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.batch_size = batch_size
        self.survival_threshold = survival_threshold
        self.seed = seed
        self.dropout_trials = max(1, dropout_trials)
        self.max_batches = max_batches
        self.allow_simulated_quant = allow_simulated_quant
        self.model_name = model_name
        self._baseline: Optional[float] = None

    @property
    def direction(self) -> str:
        return "higher_better" if self.metric == "accuracy" else "lower_better"

    def baseline(self) -> float:
        """Clean score of the uncompressed model (computed once, then cached)."""
        if self._baseline is None:
            self._baseline = self._eval(self.model)
        return self._baseline

    def run(self, methods=DEFAULT_METHODS) -> RobustnessReport:
        base = self.baseline()
        results = []
        for method in methods:
            t0 = time.time()
            try:
                after, std, backend, note = self._run_method(method)
            except _SkipMethod as e:
                results.append(MethodRobustness(
                    method=method, baseline=base, after=float("nan"), survived=False,
                    time_seconds=time.time() - t0, skipped=True, note=str(e),
                    metric_direction=self.direction))
                logger.warning(f"Robustness method '{method}' skipped: {e}")
                continue
            except Exception as e:  # noqa: BLE001 — one method failing shouldn't sink the run
                results.append(MethodRobustness(
                    method=method, baseline=base, after=float("nan"), survived=False,
                    time_seconds=time.time() - t0, skipped=True,
                    note=f"failed: {e}", metric_direction=self.direction))
                logger.warning(f"Robustness method '{method}' failed: {e}")
                continue
            # Tolerance so a degradation that lands exactly on the threshold is not
            # flipped to FAIL by float representation error.
            survived = (_degradation(base, after, self.direction)
                        <= self.survival_threshold + 1e-9)
            results.append(MethodRobustness(
                method=method, baseline=base, after=after, survived=bool(survived),
                time_seconds=time.time() - t0, std=std, backend=backend, note=note,
                metric_direction=self.direction))
        return RobustnessReport(results=results, metric=self.metric,
                                survival_threshold=self.survival_threshold,
                                model_name=self.model_name)

    # ------------------------------------------------------------------ internals
    def _eval(self, model) -> float:
        return _evaluate(model, self.eval_dataset, self.metric, self.batch_size,
                         self.max_batches)

    def _run_method(self, method: str):
        """Returns (after_score, std_or_None, backend_or_None, note)."""
        if method == "int8":
            return self._eval(_quantize_int8(self.model)), None, "torch-dynamic-int8", ""
        if method == "int4":
            return self._int4()
        m = _HEAD_PRUNING_RE.match(method)
        if m:
            return self._head_pruning(int(m.group(1)))
        m = _NEURON_DROPOUT_RE.match(method)
        if m:
            return self._neuron_dropout(int(m.group(1)))
        raise ValueError(
            f"Unknown method '{method}'. Known: int8, int4, "
            f"head_pruning_<pct>, neuron_dropout_<pct>")

    def _int4(self):
        try:
            import bitsandbytes  # noqa: F401
            has_bnb = True
        except ImportError:
            has_bnb = False
        if has_bnb:
            try:
                return self._eval(_quantize_int4_bnb(self.model)), None, "bitsandbytes-nf4", ""
            except Exception as e:  # noqa: BLE001 — fall back rather than lose the row
                logger.warning(f"bitsandbytes 4-bit unavailable ({e}); "
                               f"falling back to simulated INT4.")
        if not self.allow_simulated_quant:
            raise _SkipMethod("bitsandbytes not available and allow_simulated_quant=False")
        note = "simulated per-channel symmetric INT4 (no bitsandbytes)"
        return self._eval(_quantize_int4_simulated(self.model)), None, "simulated-int4", note

    def _head_pruning(self, pct: int):
        if not 0 < pct < 100:
            raise ValueError(f"head_pruning percentage must be in (0, 100), got {pct}")
        with _PrunedHeads(self.model, pct / 100.0, self.seed):
            return self._eval(self.model), None, None, ""

    def _neuron_dropout(self, pct: int):
        if not 0 <= pct < 100:
            raise ValueError(f"neuron_dropout percentage must be in [0, 100), got {pct}")
        modules = _ffn_expansion_linears(self.model)
        if not modules:
            raise _SkipMethod("no feed-forward expansion projections found")
        scores = []
        for t in range(self.dropout_trials):
            with _NeuronDropout(modules, pct / 100.0, self.seed + t):
                scores.append(self._eval(self.model))
        note = f"{self.dropout_trials} trial(s) over {len(modules)} FFN projection(s)"
        std = float(np.std(scores)) if len(scores) > 1 else 0.0
        return float(np.mean(scores)), std, None, note


class _SkipMethod(Exception):
    """A method could not run in this environment; record it as skipped, not failed."""


# ----------------------------------------------------------- SAL vs baseline
@dataclass
class MethodComparison:
    method: str
    base: Optional[MethodRobustness]
    sal: Optional[MethodRobustness]
    winner: str          # "SAL" | "baseline" | "tie" | "n/a"


@dataclass
class RobustnessComparison:
    """Head-to-head robustness of a SAL-trained model against a standard one."""
    rows: list
    metric: str
    baseline_report: RobustnessReport
    sal_report: RobustnessReport

    @property
    def comparable(self) -> list:
        return [r for r in self.rows if r.winner != "n/a"]

    @property
    def sal_wins(self) -> int:
        return len([r for r in self.comparable if r.winner == "SAL"])

    @property
    def table(self) -> str:
        head = (f"{'method':<18}{'base_after':>12}{'base_deg':>10}"
                f"{'sal_after':>12}{'sal_deg':>10}{'winner':>10}")
        lines = [head, "-" * len(head)]
        for r in self.rows:
            if r.winner == "n/a":
                lines.append(f"{r.method:<18}{'-':>12}{'-':>10}{'-':>12}{'-':>10}{'n/a':>10}")
                continue
            lines.append(f"{r.method:<18}{r.base.after:>12.4f}{r.base.degradation:>10.2%}"
                         f"{r.sal.after:>12.4f}{r.sal.degradation:>10.2%}{r.winner:>10}")
        lines.append("-" * len(head))
        lines.append(f"robustness_score  baseline={self.baseline_report.robustness_score:.4f}  "
                     f"SAL={self.sal_report.robustness_score:.4f}")
        return "\n".join(lines)

    @property
    def summary(self) -> str:
        n = len(self.comparable)
        if n == 0:
            return "No method ran on both models — nothing to compare."
        wins = self.sal_wins
        verb = "more robust" if wins * 2 > n else ("less robust" if wins * 2 < n else "tied")
        return (f"SAL-trained model is {verb} in {wins}/{n} compression methods "
                f"(robustness {self.sal_report.robustness_score:.3f} vs "
                f"{self.baseline_report.robustness_score:.3f}).")

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "summary": self.summary,
            "sal_wins": self.sal_wins,
            "methods_compared": len(self.comparable),
            "rows": [
                {"method": r.method, "winner": r.winner,
                 "baseline": None if r.base is None else {
                     "after": round(r.base.after, 6), "delta": round(r.base.delta, 6),
                     "degradation": round(r.base.degradation, 6),
                     "survived": bool(r.base.survived)},
                 "sal": None if r.sal is None else {
                     "after": round(r.sal.after, 6), "delta": round(r.sal.delta, 6),
                     "degradation": round(r.sal.degradation, 6),
                     "survived": bool(r.sal.survived)}}
                for r in self.rows
            ],
            "baseline_report": self.baseline_report.to_dict(),
            "sal_report": self.sal_report.to_dict(),
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_robustness_comparison_pdf
            render_robustness_comparison_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")


def robustness_compare(sal_model, baseline_model, eval_dataset,
                       methods=DEFAULT_METHODS, metric: str = "accuracy",
                       **kwargs) -> RobustnessComparison:
    """Compare how well a SAL-trained model and a standard model survive compression.

    Each model is scored against its **own** clean baseline, so the comparison is
    about resilience, not about which model was better to begin with. The winner
    of a row is the model with the lower relative degradation.

    Extra keyword arguments (``batch_size``, ``survival_threshold``, ``seed``,
    ``dropout_trials``, ``max_batches``, ``allow_simulated_quant``) are forwarded
    to both :class:`RobustnessTest` instances.
    """
    base_report = RobustnessTest(baseline_model, eval_dataset, metric=metric,
                                 model_name="baseline", **kwargs).run(methods)
    sal_report = RobustnessTest(sal_model, eval_dataset, metric=metric,
                                model_name="SAL", **kwargs).run(methods)

    by_method = {r.method: r for r in base_report.results}
    sal_by_method = {r.method: r for r in sal_report.results}

    rows = []
    for method in methods:
        b, s = by_method.get(method), sal_by_method.get(method)
        if b is None or s is None or b.skipped or s.skipped:
            rows.append(MethodComparison(method=method, base=b, sal=s, winner="n/a"))
            continue
        if abs(s.degradation - b.degradation) < 1e-9:
            winner = "tie"
        else:
            winner = "SAL" if s.degradation < b.degradation else "baseline"
        rows.append(MethodComparison(method=method, base=b, sal=s, winner=winner))

    return RobustnessComparison(rows=rows, metric=metric, baseline_report=base_report,
                                sal_report=sal_report)
