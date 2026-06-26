"""sal.compare() — benchmark SAL against common pruning methods.

Trains/compresses the same model with several methods at a matched compression
level and reports which one keeps the most accuracy (or lowest loss) after the
heads are removed. Ships two post-hoc baselines (magnitude, random) plus the
real SAL training path; custom methods can be registered.

    from sal import compare
    results = compare(model, dataset, eval_dataset,
                      methods=["sal", "magnitude", "random_posthoc"],
                      compression=0.33, sal_epochs=3, metric="accuracy")
    print(results.table)
    print(results.winner)
    results.save("comparison.json")
"""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- result types
@dataclass
class MethodResult:
    method: str
    score: float
    pruned_heads: int
    time_seconds: float


@dataclass
class CompareResult:
    results: list
    metric: str
    compression: float

    @property
    def winner(self) -> str:
        if not self.results:
            return ""
        best = max if self.metric == "accuracy" else min
        return best(self.results, key=lambda r: r.score).method

    @property
    def table(self) -> str:
        head = f"{'method':<16}{self.metric:>12}{'pruned_heads':>14}{'time_s':>10}"
        lines = [head, "-" * len(head)]
        for r in self.results:
            mark = "  <-- best" if r.method == self.winner else ""
            lines.append(f"{r.method:<16}{r.score:>12.4f}{r.pruned_heads:>14}{r.time_seconds:>10.2f}{mark}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "compression": self.compression,
            "winner": self.winner,
            "results": [{"method": r.method, "score": round(r.score, 6),
                         "pruned_heads": r.pruned_heads, "time_seconds": round(r.time_seconds, 3)}
                        for r in self.results],
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_comparison_pdf
            render_comparison_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")


# ------------------------------------------------------------- plugin registry
_CUSTOM_METHODS: dict = {}


def register_method(name: str, fn):
    """Register a custom pruning method.

    ``fn(model, dataset, eval_dataset, ctx) -> float`` returns the metric score of
    the compressed model. ``ctx`` carries: compression, metric, batch_size,
    n_prune, total_heads, num_heads_per_layer, and the eval/mask helpers.
    """
    _CUSTOM_METHODS[name] = fn


# --------------------------------------------------------------- masking helper
class _MaskedHeads:
    """Context manager that zeroes a fixed set of (layer, head) at the attention
    output projection input — the same mechanism the masker uses, kept local so
    compare() never mutates weights for the post-hoc baselines."""

    def __init__(self, model, pruned_set, num_heads):
        from sal import arch_support
        self.projs = arch_support.get_output_projections(model)
        self.nh = num_heads
        self.by_layer: dict = {i: set() for i in range(len(self.projs))}
        for (l, h) in pruned_set:
            if l in self.by_layer:
                self.by_layer[l].add(h)
        self.handles: list = []

    def _make(self, heads):
        nh = self.nh

        def hook(mod, inputs):
            x = inputs[0]
            bs, sl, hidden = x.shape
            hd = hidden // nh
            x = x.view(bs, sl, nh, hd).clone()
            for h in heads:
                x[:, :, h, :] = 0.0
            return (x.view(bs, sl, hidden),) + tuple(inputs[1:])
        return hook

    def __enter__(self):
        for li, proj in enumerate(self.projs):
            heads = self.by_layer.get(li)
            if heads:
                self.handles.append(proj.register_forward_pre_hook(self._make(heads)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------- evaluation
def _eval(model, eval_dataset, metric, batch_size):
    from sal.fi import _iter_data, _to_dev
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in _iter_data(eval_dataset, batch_size):
            batch = _to_dev(batch, device)
            out = model(**batch) if isinstance(batch, dict) else model(batch)
            if metric == "accuracy":
                logits = out.logits if hasattr(out, "logits") else out
                preds = logits.argmax(dim=-1)
                labels = batch["labels"]
                correct += (preds == labels).sum().item()
                total += labels.numel()
            else:  # loss
                loss = out.loss if hasattr(out, "loss") else out
                loss_sum += float(loss)
                n += 1
    return correct / max(total, 1) if metric == "accuracy" else loss_sum / max(n, 1)


def _head_magnitudes(model, num_heads, hidden):
    """Per-head importance = L2 norm of its slice of the output-projection weight."""
    from sal import arch_support
    hd = hidden // num_heads
    mags = {}
    for li, proj in enumerate(arch_support.get_output_projections(model)):
        w = proj.weight.detach().abs().float()
        for h in range(num_heads):
            sl = slice(h * hd, (h + 1) * hd)
            if w.shape[1] == hidden:        # nn.Linear [out, in] -> head on input columns
                block = w[:, sl]
            elif w.shape[0] == hidden:      # Conv1D [in, out] -> head on input rows
                block = w[sl, :]
            else:
                block = w
            mags[(li, h)] = float(block.norm())
    return mags


def _infer_hidden(model):
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "dim"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return v
    raise ValueError("Cannot infer hidden size from model.config")


# --------------------------------------------------------------- built-in methods
def _method_magnitude(model, dataset, eval_dataset, ctx):
    mags = _head_magnitudes(model, ctx["num_heads_per_layer"], ctx["hidden"])
    pruned = [hh for hh, _ in sorted(mags.items(), key=lambda kv: kv[1])[:ctx["n_prune"]]]
    with _MaskedHeads(model, pruned, ctx["num_heads_per_layer"]):
        return _eval(model, eval_dataset, ctx["metric"], ctx["batch_size"])


def _method_random_posthoc(model, dataset, eval_dataset, ctx):
    rng = np.random.RandomState(ctx["seed"])
    all_heads = [(l, h) for l in range(ctx["num_layers"]) for h in range(ctx["num_heads_per_layer"])]
    idx = rng.choice(len(all_heads), size=ctx["n_prune"], replace=False)
    pruned = [all_heads[i] for i in idx]
    with _MaskedHeads(model, pruned, ctx["num_heads_per_layer"]):
        return _eval(model, eval_dataset, ctx["metric"], ctx["batch_size"])


def _method_sal(model, dataset, eval_dataset, ctx):
    """Train a copy with progressive head silencing, then evaluate it WITH its
    adapted pruned set still active (that is the compressed SAL model)."""
    from torch.optim import AdamW
    from sal.config import SALConfig
    from sal.masker import HeadMasker
    from sal.fi import _iter_data, _to_dev

    m = copy.deepcopy(model)
    m.train()
    device = next(m.parameters()).device
    config = SALConfig(num_layers=ctx["num_layers"], num_heads_per_layer=ctx["num_heads_per_layer"],
                       prune_fraction=ctx["compression"])
    masker = HeadMasker(m, config, seed=ctx["seed"])
    masker.install()
    opt = AdamW(m.parameters(), lr=ctx["lr"])
    batches = list(_iter_data(dataset, ctx["batch_size"]))
    total_steps = max(1, len(batches) * ctx["sal_epochs"])
    step = 0
    for _ in range(ctx["sal_epochs"]):
        for batch in batches:
            batch = _to_dev(batch, device)
            masker.step(step, total_steps)
            out = m(**batch) if isinstance(batch, dict) else m(batch)
            loss = out.loss if hasattr(out, "loss") else out
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
    score = _eval(m, eval_dataset, ctx["metric"], ctx["batch_size"])  # masker still active
    masker.remove()
    return score


_BUILTIN = {
    "sal": _method_sal,
    "magnitude": _method_magnitude,
    "random_posthoc": _method_random_posthoc,
}


# -------------------------------------------------------------------- entrypoint
def compare(model, dataset, eval_dataset=None, methods=("sal", "magnitude", "random_posthoc"),
            compression: float = 0.33, sal_epochs: int = 3, metric: str = "accuracy",
            batch_size: int = 8, lr: float = 5e-5, seed: int = 0) -> CompareResult:
    """Benchmark pruning methods on ``model`` at a matched compression level."""
    if metric not in ("accuracy", "loss"):
        raise ValueError(f"metric must be 'accuracy' or 'loss', got '{metric}'")
    from sal.fi import _infer_num_heads
    from sal import arch_support

    eval_dataset = eval_dataset if eval_dataset is not None else dataset
    nh = _infer_num_heads(model)
    nl = len(arch_support.get_output_projections(model))
    total_heads = nl * nh
    n_prune = int(round(compression * total_heads))
    ctx = {"compression": compression, "metric": metric, "batch_size": batch_size,
           "n_prune": n_prune, "total_heads": total_heads, "num_layers": nl,
           "num_heads_per_layer": nh, "hidden": _infer_hidden(model),
           "sal_epochs": sal_epochs, "lr": lr, "seed": seed}

    results = []
    for method in methods:
        fn = _BUILTIN.get(method) or _CUSTOM_METHODS.get(method)
        if fn is None:
            raise ValueError(f"Unknown method '{method}'. Built-in: {list(_BUILTIN)}; "
                             f"registered: {list(_CUSTOM_METHODS)}")
        t0 = time.time()
        try:
            score = fn(model, dataset, eval_dataset, ctx)
        except Exception as e:  # noqa: BLE001 — one method failing shouldn't sink the rest
            logger.warning(f"Method '{method}' failed: {e}")
            score = float("nan")
        results.append(MethodResult(method=method, score=float(score),
                                    pruned_heads=n_prune, time_seconds=time.time() - t0))
    return CompareResult(results=results, metric=metric, compression=compression)


compare.register_method = register_method  # type: ignore[attr-defined]
