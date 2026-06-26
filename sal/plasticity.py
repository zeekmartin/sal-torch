"""PlasticityScanner — where can a model absorb compression?

FI tells you how fragile a model *is*. PlasticityScanner tells you how much
capacity it *has* to reorganize, so you know where it is safe to compress before
you touch anything. It measures three complementary axes and combines them into
an actionable per-layer absorption map.

  * **Routing flexibility** — attention routing entropy per layer. Many spread-out
    paths (high entropy) means information can re-route if heads are removed.
  * **Inter-layer redundancy (CKA)** — how similar adjacent layers'
    representations are. If two layers are very similar, one can absorb the
    other's work.
  * **Intra-layer redundancy (MI)** — how much heads within a layer share
    function. Redundant heads can compensate for each other.

The absorption map labels each layer ELASTIC (safe to compress), SATURATED
(structural bottleneck, leave alone), or HUB (compensates when others are
pruned). `recommend()` turns this into concrete prune / never-touch lists.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ELASTIC = "ELASTIC"
SATURATED = "SATURATED"
HUB = "HUB"

# A layer is flagged a HUB if its routing entropy rises by at least this much
# (absolute, on the 0-1 normalized scale) when heads elsewhere are masked off.
_HUB_DELTA = 0.01


# ----------------------------------------------------------------- axis helpers
def _attention_entropy(attn: torch.Tensor) -> np.ndarray:
    """Per-head attention entropy, normalized to [0, 1]. attn: [B, H, S, S]."""
    p = attn.clamp(min=0)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)      # [B, H, S] entropy per query
    ent = ent.mean(dim=(0, 2))                       # [H] mean over batch + queries
    norm = math.log(attn.shape[-1]) if attn.shape[-1] > 1 else 1.0
    return (ent / norm).clamp(0, 1).cpu().numpy()


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two representation matrices [N, d]. Returns [0, 1]."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    yx = y.T @ x
    xx = x.T @ x
    yy = y.T @ y
    den = math.sqrt(float((xx ** 2).sum()) * float((yy ** 2).sum())) + 1e-12
    return float((yx ** 2).sum() / den)


def _mean_abs_offdiag_corr(mat: np.ndarray) -> float:
    """Mean absolute pairwise Pearson correlation between rows of ``mat``.

    Used as a correlation-based proxy for mutual information between heads.
    """
    from sal.fi import _pearson_similarity
    n = mat.shape[0]
    if n < 2:
        return 0.0
    c = np.abs(_pearson_similarity(mat))
    np.fill_diagonal(c, 0.0)
    return float(c.sum() / (n * (n - 1)))


def _pool(hidden: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Masked mean over the sequence dimension. hidden: [B, S, D] -> [B, D]."""
    if attn_mask is not None:
        m = attn_mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return hidden.mean(dim=1)


# --------------------------------------------------------------- result objects
@dataclass
class Recommendation:
    target_compression: float
    safe_to_prune: list
    never_touch: list
    expected_impact: float  # estimated accuracy delta (negative = drop); heuristic

    def to_dict(self) -> dict:
        return {
            "target_compression": self.target_compression,
            "safe_to_prune": [list(h) for h in self.safe_to_prune],
            "never_touch": [list(h) for h in self.never_touch],
            "expected_impact": self.expected_impact,
        }


@dataclass
class PlasticityMap:
    routing: dict           # {layer: routing_score in [0,1]}
    cka_similarity: dict     # {(layer_i, layer_i+1): cka in [0,1]}
    mutual_info: dict        # {layer: mi_proxy in [0,1]}
    absorption_map: dict     # {layer: ELASTIC | SATURATED | HUB}
    num_layers: int = 0
    num_heads_per_layer: int = 0

    @property
    def hub_layers(self) -> list:
        return [l for l, c in self.absorption_map.items() if c == HUB]

    @property
    def elastic_layers(self) -> list:
        return [l for l, c in self.absorption_map.items() if c == ELASTIC]

    @property
    def saturated_layers(self) -> list:
        return [l for l, c in self.absorption_map.items() if c == SATURATED]

    @property
    def summary(self) -> str:
        return (f"{len(self.elastic_layers)} elastic, {len(self.saturated_layers)} saturated, "
                f"{len(self.hub_layers)} hub  |  mean routing="
                f"{np.mean(list(self.routing.values())):.3f}, mean MI="
                f"{np.mean(list(self.mutual_info.values())):.3f}")

    def recommend(self, target_compression: float = 0.33) -> Recommendation:
        """Turn the absorption map into prune / never-touch recommendations."""
        nh = self.num_heads_per_layer
        total = self.num_layers * nh
        n_target = int(round(target_compression * total))

        # Never touch structural hubs.
        never = [(l, h) for l in self.hub_layers for h in range(nh)]

        # Prefer pruning elastic layers (most redundant first), then saturated.
        elastic = sorted(self.elastic_layers, key=lambda l: self.mutual_info.get(l, 0), reverse=True)
        saturated = sorted(self.saturated_layers, key=lambda l: self.mutual_info.get(l, 0), reverse=True)

        safe, n_elastic, n_saturated = [], 0, 0
        for layer in elastic + saturated:
            for h in range(nh):
                if len(safe) >= n_target:
                    break
                safe.append((layer, h))
                if layer in self.elastic_layers:
                    n_elastic += 1
                else:
                    n_saturated += 1
            if len(safe) >= n_target:
                break

        # Heuristic impact estimate: elastic heads are nearly free, saturated cost more.
        expected_impact = -(0.002 * n_elastic + 0.02 * n_saturated)
        return Recommendation(target_compression=target_compression, safe_to_prune=safe,
                              never_touch=never, expected_impact=round(expected_impact, 4))

    def to_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_heads_per_layer": self.num_heads_per_layer,
            "routing": {str(k): round(v, 4) for k, v in self.routing.items()},
            "cka_similarity": {f"{a}-{b}": round(v, 4) for (a, b), v in self.cka_similarity.items()},
            "mutual_info": {str(k): round(v, 4) for k, v in self.mutual_info.items()},
            "absorption_map": {str(k): v for k, v in self.absorption_map.items()},
            "hub_layers": self.hub_layers,
            "summary": self.summary,
        }

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_plasticity_pdf
            render_plasticity_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")


# ------------------------------------------------------------------- the scanner
class PlasticityScanner:
    def __init__(self, model: nn.Module, probe_dataset, num_samples: int = 200, batch_size: int = 16):
        self.model = model
        self.probe_dataset = probe_dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

    # -- data plumbing (kept local so plasticity is independent of fi/masker) --
    def _iter(self):
        from sal.fi import _iter_data
        return _iter_data(self.probe_dataset, self.batch_size)

    def _to_dev(self, batch, device):
        from sal.fi import _to_dev
        return _to_dev(batch, device)

    def _num_heads(self) -> int:
        from sal.fi import _infer_num_heads
        return _infer_num_heads(self.model)

    def scan(self) -> PlasticityMap:
        from sal import arch_support
        self.model.eval()
        device = next(self.model.parameters()).device
        nh = self._num_heads()
        out_projs = arch_support.get_output_projections(self.model)
        nl = len(out_projs)

        # Pre-hooks capture the per-head signatures for the MI axis (same point
        # the FI graph uses: the input to each attention output projection).
        captures: list = [None] * nl

        def make_hook(i):
            def fn(mod, inputs):
                captures[i] = inputs[0].detach()
            return fn

        hooks = [p.register_forward_pre_hook(make_hook(i)) for i, p in enumerate(out_projs)]

        routing_acc: list = [[] for _ in range(nl)]   # per layer: list of [H] entropy
        cka_acc: list = [[] for _ in range(nl)]        # per layer: list of [B, D] pooled reps
        mi_acc: list = [[] for _ in range(nl)]         # per layer: list of [B, nh, head_dim]
        attentions_ok = True

        try:
            seen = 0
            with torch.no_grad():
                for batch in self._iter():
                    batch = self._to_dev(batch, device)
                    attn_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
                    out = self._forward(batch)
                    bs = self._batch_size(batch)

                    atts = getattr(out, "attentions", None)
                    hids = getattr(out, "hidden_states", None)
                    if atts is None:
                        attentions_ok = False
                    else:
                        for li in range(min(nl, len(atts))):
                            if atts[li] is not None:
                                routing_acc[li].append(_attention_entropy(atts[li]))
                    if hids is not None:
                        layer_hids = hids[1:] if len(hids) == nl + 1 else hids
                        for li in range(min(nl, len(layer_hids))):
                            cka_acc[li].append(_pool(layer_hids[li], attn_mask).cpu().numpy())

                    # MI signatures from captured projection inputs.
                    for li in range(nl):
                        x = captures[li]
                        if x is None:
                            continue
                        pooled = _pool(x, attn_mask)             # [B, D]
                        mi_acc[li].append(pooled.view(pooled.shape[0], nh, -1).cpu().numpy())
                        captures[li] = None

                    seen += bs
                    if seen >= self.num_samples:
                        break
        finally:
            for h in hooks:
                h.remove()

        routing = self._routing_scores(routing_acc, nl, attentions_ok)
        cka = self._cka_scores(cka_acc, nl)
        mutual_info = self._mi_scores(mi_acc, nl)
        hubs = self._detect_hubs(routing, device, nl, nh)
        absorption = self._absorption_map(routing, cka, mutual_info, hubs, nl)

        return PlasticityMap(routing=routing, cka_similarity=cka, mutual_info=mutual_info,
                             absorption_map=absorption, num_layers=nl, num_heads_per_layer=nh)

    # ----------------------------------------------------------- forward helpers
    def _forward(self, batch):
        kw = dict(output_attentions=True, output_hidden_states=True)
        if isinstance(batch, dict):
            try:
                return self.model(**batch, **kw)
            except TypeError:
                return self.model(**batch)
        try:
            return self.model(batch, **kw)
        except TypeError:
            return self.model(batch)

    @staticmethod
    def _batch_size(batch) -> int:
        if isinstance(batch, dict):
            v = next(iter(batch.values()))
            return v.shape[0] if hasattr(v, "shape") else 1
        return batch.shape[0] if hasattr(batch, "shape") else 1

    # ------------------------------------------------------------- score builders
    @staticmethod
    def _routing_scores(routing_acc, nl, attentions_ok) -> dict:
        out = {}
        for li in range(nl):
            if routing_acc[li]:
                out[li] = float(np.mean(np.concatenate([r[None, :] for r in routing_acc[li]], 0)))
            else:
                out[li] = float("nan")
        if not attentions_ok:
            logger.warning("Model did not return attention weights; routing scores are "
                           "unavailable. Load with attn_implementation='eager' to enable them.")
        return out

    @staticmethod
    def _cka_scores(cka_acc, nl) -> dict:
        reps = [np.concatenate(c, 0) if c else None for c in cka_acc]
        out = {}
        for li in range(nl - 1):
            if reps[li] is not None and reps[li + 1] is not None:
                out[(li, li + 1)] = _linear_cka(reps[li], reps[li + 1])
        return out

    @staticmethod
    def _mi_scores(mi_acc, nl) -> dict:
        out = {}
        for li in range(nl):
            if not mi_acc[li]:
                out[li] = 0.0
                continue
            stacked = np.concatenate(mi_acc[li], 0)          # [N, nh, head_dim]
            n, heads, hd = stacked.shape
            per_head = stacked.transpose(1, 0, 2).reshape(heads, n * hd)  # [nh, N*head_dim]
            out[li] = _mean_abs_offdiag_corr(per_head)
        return out

    def _detect_hubs(self, baseline_routing: dict, device, nl: int, nh: int) -> list:
        """A hub is a layer whose routing entropy *rises* when heads elsewhere are
        masked off — i.e. it picks up the slack. Uses the existing HeadMasker."""
        if any(math.isnan(v) for v in baseline_routing.values()):
            return []  # routing unavailable -> cannot detect hubs this way
        try:
            from sal.config import SALConfig
            from sal.masker import HeadMasker
            config = SALConfig(num_layers=nl, num_heads_per_layer=nh, prune_fraction=0.33)
            masker = HeadMasker(self.model, config, seed=0)
            masker.install()
            masker.activate()  # silence a random ~33% of heads across the model
        except Exception as e:  # noqa: BLE001 — hub detection is best-effort
            logger.warning(f"Hub detection skipped: {e}")
            return []

        masked_routing = {li: [] for li in range(nl)}
        try:
            seen = 0
            with torch.no_grad():
                for batch in self._iter():
                    batch = self._to_dev(batch, device)
                    out = self._forward(batch)
                    atts = getattr(out, "attentions", None)
                    if atts is None:
                        break
                    for li in range(min(nl, len(atts))):
                        if atts[li] is not None:
                            masked_routing[li].append(float(np.mean(_attention_entropy(atts[li]))))
                    seen += self._batch_size(batch)
                    if seen >= min(self.num_samples, 64):
                        break
        finally:
            masker.remove()

        hubs = []
        for li in range(nl):
            if masked_routing[li]:
                delta = float(np.mean(masked_routing[li])) - baseline_routing[li]
                if delta > _HUB_DELTA:
                    hubs.append(li)
        return hubs

    @staticmethod
    def _absorption_map(routing, cka, mutual_info, hubs, nl) -> dict:
        # Per-layer CKA = best redundancy with a neighbour.
        cka_layer = {}
        for li in range(nl):
            vals = [cka[k] for k in cka if li in k]
            cka_layer[li] = max(vals) if vals else 0.0

        def _median(d):
            vals = [v for v in d.values() if not (isinstance(v, float) and math.isnan(v))]
            return float(np.median(vals)) if vals else 0.0

        r_med, c_med, m_med = _median(routing), _median(cka_layer), _median(mutual_info)

        amap = {}
        for li in range(nl):
            if li in hubs:
                amap[li] = HUB
                continue
            r = routing.get(li, float("nan"))
            r_high = (not math.isnan(r) and r >= r_med) or math.isnan(r)  # ignore axis if unavailable
            c_high = cka_layer.get(li, 0.0) >= c_med
            m_high = mutual_info.get(li, 0.0) >= m_med
            amap[li] = ELASTIC if (r_high and c_high and m_high) else SATURATED
        return amap
