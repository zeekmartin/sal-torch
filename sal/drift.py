"""DriftMonitor — measure structural forgetting after fine-tuning.

After a model is fine-tuned on a new task, has it quietly reorganized the
structure that carried the old one? DriftMonitor answers this by taking a
**structural snapshot** before and after, then comparing them.

A snapshot captures, from a fixed probe set:

  * the model's fragility score (FI) and per-layer classification,
  * a per-layer activation representation (used for CKA drift), from which
    per-head signatures are derived.

Comparing two snapshots produces a :class:`DriftReport`:

  * ``forgetting_score``    — overall structural change, in [0, 1].
  * ``structural_delta``    — FI_after - FI_before (positive = more fragile).
  * ``layer_drift``         — per-layer activation CKA (1 = identical).
  * ``protected_integrity`` — for a known protected head set, how well those
                              heads' activations held (near 1 = protection worked).
  * ``classification_changes`` — layers whose class flipped (e.g. IMMUNE→CRITICAL).

DriftMonitor is independent of StructuralGuard: it measures forgetting after
*any* fine-tuning, guarded or not.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sal.fi import (DEFAULT_EDGE_DENSITY, classify_layers, compute_fi, _infer_num_heads,
                    _iter_data, _pearson_similarity, _threshold_for_density, _to_dev)
from sal.plasticity import _linear_cka

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- internals
def _pool(hidden: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Masked mean over the sequence dimension. hidden: [B, S, D] -> [B, D]."""
    if attn_mask is not None:
        m = attn_mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return hidden.mean(dim=1)


def _capture_reps(model, probe_dataset, num_samples: int, batch_size: int):
    """Per-layer pooled activation representations, captured at the attention
    output-projection input (the same point FI and plasticity use)."""
    from sal import arch_support
    model.eval()
    device = next(model.parameters()).device
    out_projs = arch_support.get_output_projections(model)
    nl = len(out_projs)
    nh = _infer_num_heads(model)
    captures: list = [None] * nl
    per_layer: list = [[] for _ in range(nl)]

    def make_hook(i):
        def fn(mod, inputs):
            captures[i] = inputs[0].detach()
        return fn

    hooks = [p.register_forward_pre_hook(make_hook(i)) for i, p in enumerate(out_projs)]
    try:
        n = 0
        with torch.no_grad():
            for batch in _iter_data(probe_dataset, batch_size):
                batch = _to_dev(batch, device)
                am = batch.get("attention_mask") if isinstance(batch, dict) else None
                model(**batch) if isinstance(batch, dict) else model(batch)
                bs = 0
                for i in range(nl):
                    x = captures[i]
                    if x is None:
                        continue
                    pooled = _pool(x, am)              # [B, hidden]
                    per_layer[i].append(pooled.cpu().numpy())
                    captures[i] = None
                    bs = pooled.shape[0]
                n += bs
                if n >= num_samples:
                    break
    finally:
        for h in hooks:
            h.remove()

    reps = [np.concatenate(p, 0)[:num_samples] if p else np.zeros((0, 0), dtype=np.float32)
            for p in per_layer]
    return reps, nl, nh


def _adjacency_from_reps(reps: list, nh: int, density: float = DEFAULT_EDGE_DENSITY) -> np.ndarray:
    """Build the binary head-adjacency graph from per-layer reps (same recipe as
    fi.extract_activation_graph, but reusing already-captured signatures)."""
    sigs = []
    for r in reps:
        if r.size == 0:
            continue
        n = r.shape[0]
        rr = r.reshape(n, nh, -1)
        for h in range(nh):
            sigs.append(rr[:, h, :].reshape(-1).astype(np.float64))
    if not sigs:
        return np.zeros((0, 0), dtype=np.int8)
    X = np.stack(sigs)
    S = _pearson_similarity(X)
    thr = _threshold_for_density(S, density)
    adj = (np.abs(S) > thr).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj


# ----------------------------------------------------------------- result types
@dataclass
class StructuralSnapshot:
    """A frozen structural state: fragility, layer classes, and activation reps."""
    name: str
    fi_score: float
    layer_map: dict            # {layer:int -> class str}
    layer_reps: list           # list of np.ndarray [N, hidden] per layer
    num_layers: int
    num_heads: int

    def head_signature(self, layer: int, head: int) -> np.ndarray:
        """Per-head activation signature [N, head_dim] for one (layer, head)."""
        r = self.layer_reps[layer]
        n = r.shape[0]
        return r.reshape(n, self.num_heads, -1)[:, head, :]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fi_score": self.fi_score,
            "layer_map": {str(k): v for k, v in self.layer_map.items()},
            "layer_reps": [r.astype(np.float32).tolist() for r in self.layer_reps],
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralSnapshot":
        return cls(
            name=d["name"],
            fi_score=float(d["fi_score"]),
            layer_map={int(k): v for k, v in d.get("layer_map", {}).items()},
            layer_reps=[np.asarray(r, dtype=np.float32) for r in d.get("layer_reps", [])],
            num_layers=int(d["num_layers"]),
            num_heads=int(d["num_heads"]),
        )


@dataclass
class DriftReport:
    forgetting_score: float
    structural_delta: float
    layer_drift: dict                       # {layer:int -> cka float in [0,1]}
    protected_integrity: Optional[float]    # mean CKA over protected heads, or None
    classification_changes: dict            # {layer:int -> (before_str, after_str)}
    fi_before: float = 0.0
    fi_after: float = 0.0
    name_before: str = ""
    name_after: str = ""

    @property
    def summary(self) -> str:
        pi = "n/a" if self.protected_integrity is None else f"{self.protected_integrity:.3f}"
        drift_vals = [v for v in self.layer_drift.values() if not _isnan(v)]
        mean_ret = float(np.mean(drift_vals)) if drift_vals else float("nan")
        return (f"forgetting={self.forgetting_score:.3f}  |  "
                f"FI {self.fi_before:.3f} -> {self.fi_after:.3f} "
                f"(delta {self.structural_delta:+.3f})  |  "
                f"mean layer retention={mean_ret:.3f}  |  "
                f"protected integrity={pi}  |  "
                f"{len(self.classification_changes)} layer(s) reclassified")

    def to_dict(self) -> dict:
        return {
            "forgetting_score": round(self.forgetting_score, 6),
            "structural_delta": round(self.structural_delta, 6),
            "fi_before": round(self.fi_before, 6),
            "fi_after": round(self.fi_after, 6),
            "layer_drift": {str(k): (None if _isnan(v) else round(float(v), 6))
                            for k, v in self.layer_drift.items()},
            "protected_integrity": (None if self.protected_integrity is None
                                    else round(self.protected_integrity, 6)),
            "classification_changes": {str(k): list(v) for k, v in self.classification_changes.items()},
            "name_before": self.name_before,
            "name_after": self.name_after,
            "summary": self.summary,
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_drift_pdf
            render_drift_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")


def _isnan(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


# ----------------------------------------------------------------- snapshotting
def take_snapshot(model, probe_dataset, name: str, num_samples: int = 200,
                  batch_size: int = 16, density: float = DEFAULT_EDGE_DENSITY) -> StructuralSnapshot:
    """Capture a structural snapshot of ``model`` on ``probe_dataset``."""
    reps, nl, nh = _capture_reps(model, probe_dataset, num_samples, batch_size)
    adj = _adjacency_from_reps(reps, nh, density)
    if adj.size:
        fi = compute_fi(adj)
        lm = classify_layers(model, adj, num_heads_per_layer=nh)
        lm = {int(k): (v.value if hasattr(v, "value") else str(v)) for k, v in lm.items()}
    else:
        fi, lm = 0.0, {}
    return StructuralSnapshot(name=name, fi_score=fi, layer_map=lm, layer_reps=reps,
                              num_layers=nl, num_heads=nh)


def compare_snapshots(before: StructuralSnapshot, after: StructuralSnapshot,
                      protected_heads: Optional[list] = None) -> DriftReport:
    """Compare two structural snapshots into a :class:`DriftReport`.

    ``protected_heads`` (a list of ``(layer, head)`` tuples) is optional; when
    provided, ``protected_integrity`` reports how well those heads held.
    """
    nl = min(before.num_layers, after.num_layers)
    nh = before.num_heads
    structural_delta = after.fi_score - before.fi_score

    # Per-layer activation CKA — 1.0 means the layer's representation is unchanged.
    layer_drift = {}
    for li in range(nl):
        a, b = before.layer_reps[li], after.layer_reps[li]
        if a.size == 0 or b.size == 0:
            layer_drift[li] = float("nan")
            continue
        n = min(a.shape[0], b.shape[0])
        layer_drift[li] = _linear_cka(a[:n], b[:n])

    # Layers whose classification flipped.
    changes = {}
    for li in range(nl):
        cb, ca = before.layer_map.get(li), after.layer_map.get(li)
        if cb is not None and ca is not None and cb != ca:
            changes[li] = (cb, ca)

    # Protected integrity — per-head activation CKA over the protected set.
    protected_integrity = None
    if protected_heads:
        cors = []
        for (li, h) in protected_heads:
            if li >= nl:
                continue
            a, b = before.layer_reps[li], after.layer_reps[li]
            if a.size == 0 or b.size == 0:
                continue
            n = min(a.shape[0], b.shape[0])
            ah = a[:n].reshape(n, nh, -1)[:, h, :]
            bh = b[:n].reshape(n, nh, -1)[:, h, :]
            cors.append(_linear_cka(ah, bh))
        protected_integrity = float(np.mean(cors)) if cors else None

    # Forgetting score — blend of activation drift (dominant), FI change, and
    # classification churn. All terms are in [0, 1]; identical states give 0.
    drift_terms = [1.0 - v for v in layer_drift.values() if not _isnan(v)]
    mean_drift = float(np.mean(drift_terms)) if drift_terms else 0.0
    fi_change = min(1.0, abs(structural_delta))
    class_frac = len(changes) / max(nl, 1)
    forgetting = float(np.clip(0.5 * mean_drift + 0.3 * fi_change + 0.2 * class_frac, 0.0, 1.0))

    return DriftReport(
        forgetting_score=forgetting,
        structural_delta=structural_delta,
        layer_drift=layer_drift,
        protected_integrity=protected_integrity,
        classification_changes=changes,
        fi_before=before.fi_score,
        fi_after=after.fi_score,
        name_before=before.name,
        name_after=after.name,
    )


# --------------------------------------------------------------------- monitor
class DriftMonitor:
    """Store structural snapshots of a model and compare any two of them.

    Snapshots are keyed by name, so drift can be tracked across many sequential
    tasks (snapshot before/after each, compare any pair).
    """

    def __init__(self, model, probe_dataset, num_samples: int = 200, batch_size: int = 16):
        self.model = model
        self.probe_dataset = probe_dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.snapshots: dict = {}

    def snapshot(self, name: str) -> StructuralSnapshot:
        snap = take_snapshot(self.model, self.probe_dataset, name,
                             self.num_samples, self.batch_size)
        self.snapshots[name] = snap
        return snap

    def compare(self, name_before: str, name_after: str,
                protected_heads: Optional[list] = None) -> DriftReport:
        for nm in (name_before, name_after):
            if nm not in self.snapshots:
                raise KeyError(f"No snapshot named '{nm}'. Have: {list(self.snapshots)}")
        return compare_snapshots(self.snapshots[name_before], self.snapshots[name_after],
                                 protected_heads=protected_heads)
