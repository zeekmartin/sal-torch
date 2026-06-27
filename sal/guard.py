"""StructuralGuard — continual learning without replay buffers.

After an initial training run, a model has a *structural map*: some components
are critical (hub layers, structural bottlenecks, functionally unique heads),
others are elastic (redundant, safe to repurpose). StructuralGuard reads that
map and protects the critical components during fine-tuning on a new task, so
the model can absorb the new domain without overwriting the old one.

This is **not** EWC, replay, or distillation. The topology itself decides what
to protect:

  * Phase 1 — **scan**. Run :class:`~sal.plasticity.PlasticityScanner` to label
    each layer ELASTIC / SATURATED / HUB, and derive a per-head redundancy
    score (how much each head's function is shared by its neighbours).
  * Phase 2 — **protect**. Freeze the critical heads by zeroing their gradients
    with backward hooks (HUB layers, SATURATED layers, and the *unique*
    low-redundancy heads in ELASTIC layers). The redundant high-redundancy
    heads stay trainable and absorb the new task.
  * Phase 3 — **verify**. After fine-tuning, :meth:`measure_drift` reports what
    changed, what was preserved, and a forgetting score.

Protection is at the **attention-head level** (the same granularity as SAL), so
some heads in a layer can be frozen while others in the same layer keep learning.
Guard state is serializable (JSON) and works with any training loop — it uses
backward hooks, not training-loop modifications.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sal import arch_support
from sal.drift import StructuralSnapshot, compare_snapshots, take_snapshot
from sal.fi import _infer_num_heads, _pearson_similarity
from sal.plasticity import ELASTIC, HUB, SATURATED, PlasticityScanner

logger = logging.getLogger(__name__)


# ----------------------------------------------------- per-head redundancy (MI)
def _per_head_mi(reps: list, nh: int) -> dict:
    """Per-head redundancy score in [0, 1] — mean absolute correlation of each
    head's signature to the other heads in its layer.

    High score = redundant (its function is shared, safe to repurpose); low
    score = functionally unique (protect it). This is the same correlation-based
    proxy the plasticity MI axis uses, kept per-head here so individual heads in
    an ELASTIC layer can be ranked.
    """
    out: dict = {}
    for li, r in enumerate(reps):
        if r.size == 0:
            for h in range(nh):
                out[(li, h)] = 0.0
            continue
        n = r.shape[0]
        per_head = r.reshape(n, nh, -1).transpose(1, 0, 2).reshape(nh, -1)  # [nh, N*head_dim]
        c = np.abs(_pearson_similarity(per_head))
        np.fill_diagonal(c, 0.0)
        denom = max(nh - 1, 1)
        for h in range(nh):
            out[(li, h)] = float(c[h].sum() / denom)
    return out


# ------------------------------------------------------- gradient-mask plumbing
def _is_linear(m) -> bool:
    return isinstance(m, nn.Linear)


def _ensure_mask(masks: dict, param: torch.Tensor) -> torch.Tensor:
    entry = masks.get(id(param))
    if entry is None:
        entry = (param, torch.ones_like(param))
        masks[id(param)] = entry
    return entry[1]


def _mask_qkv(qkv: dict, h: int, nh: int, masks: dict):
    """Zero the Q/K/V weight (and bias) slices that produce head ``h``."""
    if qkv["mode"] == "separate":
        for m in (qkv["q"], qkv["k"], qkv["v"]):
            if _is_linear(m):                       # weight [out, in]; head on output rows
                out_dim = m.weight.shape[0]
                hd = out_dim // nh
                lo, hi = h * hd, (h + 1) * hd
                _ensure_mask(masks, m.weight)[lo:hi, :] = 0.0
            else:                                   # Conv1D weight [in, out]; head on out cols
                out_dim = m.weight.shape[1]
                hd = out_dim // nh
                lo, hi = h * hd, (h + 1) * hd
                _ensure_mask(masks, m.weight)[:, lo:hi] = 0.0
            b = getattr(m, "bias", None)
            if b is not None:
                _ensure_mask(masks, b)[lo:hi] = 0.0
    else:                                           # fused Q|K|V in one projection
        m = qkv["qkv"]
        if _is_linear(m):
            out_dim = m.weight.shape[0]             # 3 * hidden
            hidden = out_dim // 3
            hd = hidden // nh
            wmask = _ensure_mask(masks, m.weight)
            bmask = _ensure_mask(masks, m.bias) if getattr(m, "bias", None) is not None else None
            for blk in range(3):
                lo = blk * hidden + h * hd
                hi = lo + hd
                wmask[lo:hi, :] = 0.0
                if bmask is not None:
                    bmask[lo:hi] = 0.0
        else:                                       # Conv1D [in, 3*hidden]; out on cols
            out_dim = m.weight.shape[1]
            hidden = out_dim // 3
            hd = hidden // nh
            wmask = _ensure_mask(masks, m.weight)
            bmask = _ensure_mask(masks, m.bias) if getattr(m, "bias", None) is not None else None
            for blk in range(3):
                lo = blk * hidden + h * hd
                hi = lo + hd
                wmask[:, lo:hi] = 0.0
                if bmask is not None:
                    bmask[lo:hi] = 0.0


def _mask_output(o, h: int, nh: int, masks: dict):
    """Zero the output-projection weight slice that consumes head ``h``."""
    if o is None:
        return
    if _is_linear(o):                               # weight [out, in]; head on input cols
        hidden = o.weight.shape[1]
        hd = hidden // nh
        _ensure_mask(masks, o.weight)[:, h * hd:(h + 1) * hd] = 0.0
    else:                                           # Conv1D [in, out]; head on input rows
        hidden = o.weight.shape[0]
        hd = hidden // nh
        _ensure_mask(masks, o.weight)[h * hd:(h + 1) * hd, :] = 0.0
    # The output-projection bias is per-output-feature (mixed across heads), so
    # it cannot be attributed to a single head — left trainable.


def _build_grad_masks(model, protected_set: list, nh: int) -> dict:
    """For every protected (layer, head), build {id(param) -> (param, mask)}
    where mask is 0 on that head's weight/bias slices and 1 elsewhere."""
    masks: dict = {}
    attn_mods = arch_support.get_attention_modules(model)
    by_layer: dict = {}
    for (l, h) in protected_set:
        by_layer.setdefault(l, []).append(h)
    for li, attn in enumerate(attn_mods):
        heads = by_layer.get(li)
        if not heads:
            continue
        o = arch_support.get_output_projection(attn)
        qkv = arch_support.get_qkv_projections(attn)
        for h in heads:
            _mask_output(o, h, nh, masks)
            if qkv is not None:
                _mask_qkv(qkv, h, nh, masks)
    return masks


# --------------------------------------------------------------------- guard
class StructuralGuard:
    """Protects critical attention heads during fine-tuning on a new task."""

    def __init__(self, protected_heads: list, trainable_heads: list, num_layers: int,
                 num_heads: int, protection_level: float, baseline: Optional[StructuralSnapshot] = None,
                 probe_dataset=None, num_samples: int = 200, batch_size: int = 16,
                 head_mi: Optional[dict] = None, absorption_map: Optional[dict] = None):
        self._protected = sorted(tuple(x) for x in protected_heads)
        self._trainable = sorted(tuple(x) for x in trainable_heads)
        self._nl = num_layers
        self._nh = num_heads
        self._level = protection_level
        self._baseline = baseline
        self._probe = probe_dataset
        self._num_samples = num_samples
        self._batch_size = batch_size
        self._head_mi = head_mi or {}
        self._amap = absorption_map or {}
        self._handles: list = []

    # ------------------------------------------------------------- construction
    @classmethod
    def from_model(cls, model, probe_dataset, protection_level: float = 0.5,
                   num_samples: int = 200, batch_size: int = 16) -> "StructuralGuard":
        """Scan ``model`` and build a guard protecting the most critical
        ``protection_level`` fraction of its attention heads."""
        if not 0.0 <= protection_level <= 1.0:
            raise ValueError(f"protection_level must be in [0, 1], got {protection_level}")
        nh = _infer_num_heads(model)
        pmap = PlasticityScanner(model, probe_dataset, num_samples=num_samples,
                                 batch_size=batch_size).scan()
        baseline = take_snapshot(model, probe_dataset, "baseline", num_samples, batch_size)
        nl = baseline.num_layers
        head_mi = _per_head_mi(baseline.layer_reps, nh)
        protected, trainable = cls._rank_and_split(pmap.absorption_map, head_mi, nl, nh, protection_level)
        return cls(protected, trainable, nl, nh, protection_level, baseline, probe_dataset,
                   num_samples, batch_size, head_mi, pmap.absorption_map)

    @staticmethod
    def _rank_and_split(absorption_map: dict, head_mi: dict, nl: int, nh: int, level: float):
        """Rank heads most-critical first, then split at the protection cutoff.

        Order: HUB layer heads, then SATURATED layer heads, then ELASTIC heads
        by ascending redundancy (unique heads before redundant ones). The top
        ``level`` fraction is protected; the rest stay trainable.
        """
        hub, sat, elastic = [], [], []
        for li in range(nl):
            cls_ = absorption_map.get(li, SATURATED)
            bucket = hub if cls_ == HUB else (elastic if cls_ == ELASTIC else sat)
            for h in range(nh):
                bucket.append((li, h))
        by_uniqueness = lambda it: head_mi.get(it, 0.0)   # low MI (unique) first
        hub.sort(key=by_uniqueness)
        sat.sort(key=by_uniqueness)
        elastic.sort(key=by_uniqueness)
        ranked = hub + sat + elastic

        total = nl * nh
        n_protected = max(0, min(total, int(round(level * total))))
        protected = sorted(ranked[:n_protected])
        trainable = sorted(ranked[n_protected:])
        return protected, trainable

    # ------------------------------------------------------------------ queries
    @property
    def protected_heads(self) -> list:
        return list(self._protected)

    @property
    def trainable_heads(self) -> list:
        return list(self._trainable)

    @property
    def protection_map(self) -> dict:
        m: dict = {}
        for (l, h) in self._protected:
            m.setdefault(l, []).append(h)
        return {l: sorted(hs) for l, hs in m.items()}

    @property
    def num_layers(self) -> int:
        return self._nl

    @property
    def num_heads(self) -> int:
        return self._nh

    @property
    def protection_level(self) -> float:
        return self._level

    @property
    def is_active(self) -> bool:
        return bool(self._handles)

    # --------------------------------------------------------- protect / release
    def protect(self, model) -> "StructuralGuard":
        """Register backward hooks that zero gradients for protected heads.

        Protected heads still participate in the forward pass; only their
        gradients are masked, so their weights are frozen in place.
        """
        self.release()
        masks = _build_grad_masks(model, self._protected, self._nh)
        for param, mask in masks.values():
            handle = param.register_hook(lambda grad, m=mask: grad * m)
            self._handles.append(handle)
        logger.info(f"StructuralGuard active: protecting {len(self._protected)} heads "
                    f"across {len(self.protection_map)} layers ({len(masks)} weight tensors).")
        return self

    def release(self, model=None) -> "StructuralGuard":
        """Remove all gradient hooks; gradients flow normally afterwards."""
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    # ------------------------------------------------------------------- verify
    def measure_drift(self, model, probe_dataset=None):
        """Compare the model's current structure against the stored baseline."""
        if self._baseline is None:
            raise RuntimeError("Guard has no baseline snapshot; create it via from_model().")
        probe = probe_dataset if probe_dataset is not None else self._probe
        if probe is None:
            raise ValueError("No probe_dataset available for drift measurement.")
        after = take_snapshot(model, probe, "after", self._num_samples, self._batch_size)
        return compare_snapshots(self._baseline, after, protected_heads=self._protected)

    # --------------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "protection_level": self._level,
            "num_layers": self._nl,
            "num_heads": self._nh,
            "protected_heads": [list(x) for x in self._protected],
            "trainable_heads": [list(x) for x in self._trainable],
            "head_mi": {f"{l},{h}": round(v, 6) for (l, h), v in self._head_mi.items()},
            "absorption_map": {str(k): v for k, v in self._amap.items()},
            "baseline": self._baseline.to_dict() if self._baseline is not None else None,
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_guard_pdf
            render_guard_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")

    @classmethod
    def load(cls, path: str) -> "StructuralGuard":
        d = json.loads(Path(path).read_text())
        head_mi = {}
        for k, v in d.get("head_mi", {}).items():
            l, h = k.split(",")
            head_mi[(int(l), int(h))] = float(v)
        amap = {int(k): v for k, v in d.get("absorption_map", {}).items()}
        baseline = StructuralSnapshot.from_dict(d["baseline"]) if d.get("baseline") else None
        return cls(
            protected_heads=[tuple(x) for x in d["protected_heads"]],
            trainable_heads=[tuple(x) for x in d["trainable_heads"]],
            num_layers=int(d["num_layers"]),
            num_heads=int(d["num_heads"]),
            protection_level=float(d["protection_level"]),
            baseline=baseline,
            head_mi=head_mi,
            absorption_map=amap,
        )


# --------------------------------------------------- HuggingFace Trainer support
try:
    from transformers import TrainerCallback
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

    class TrainerCallback:  # type: ignore
        pass


class StructuralGuardCallback(TrainerCallback):
    """Auto-apply a :class:`StructuralGuard` over a HuggingFace ``Trainer`` run.

    Protection is applied on ``train_begin``, removed on ``train_end``, and the
    resulting :class:`~sal.drift.DriftReport` is stored on ``drift_report``.
    """

    def __init__(self, guard: StructuralGuard, measure_drift: bool = True):
        if not _HAS_TF:
            raise ImportError("StructuralGuardCallback requires transformers. "
                              "pip install sal-torch[hf]")
        self.guard = guard
        self.measure_drift = measure_drift
        self.drift_report = None

    def on_train_begin(self, args, state, control, model=None, **kw):
        if model is None:
            raise RuntimeError("StructuralGuardCallback needs the model.")
        self.guard.protect(model)

    def on_train_end(self, args, state, control, model=None, **kw):
        self.guard.release()
        if self.measure_drift and model is not None and self.guard._baseline is not None:
            try:
                self.drift_report = self.guard.measure_drift(model)
            except Exception as e:  # noqa: BLE001 — drift measurement is best-effort
                logger.warning(f"Drift measurement skipped: {e}")
