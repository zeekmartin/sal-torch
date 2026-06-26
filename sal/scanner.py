"""High-level structural analysis: FIScanner and FIMonitor."""
from __future__ import annotations
import json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from sal.fi import LayerClass, classify_layers, compute_fi, extract_activation_graph

logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    fi_score: float
    layer_map: dict[int, LayerClass]
    adjacency: np.ndarray
    num_layers: int = 0
    num_heads_per_layer: int = 0

    @property
    def critical_layers(self): return [i for i,c in self.layer_map.items() if c == LayerClass.CRITICAL]
    @property
    def immune_layers(self): return [i for i,c in self.layer_map.items() if c == LayerClass.IMMUNE]
    @property
    def buffer_layers(self): return [i for i,c in self.layer_map.items() if c == LayerClass.BUFFER]
    @property
    def summary(self):
        return f"FI={self.fi_score:.4f} | {len(self.immune_layers)} immune, {len(self.buffer_layers)} buffer, {len(self.critical_layers)} critical"

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            data = {"fi_score": self.fi_score, "num_layers": self.num_layers,
                    "layer_classification": {str(k): v.value for k,v in self.layer_map.items()}}
            p.write_text(json.dumps(data, indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_fi_pdf
            render_fi_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")

class FIScanner:
    def __init__(self, model, probe_dataset, num_samples=500, batch_size=16):
        self.model = model; self.probe_dataset = probe_dataset
        self.num_samples = num_samples; self.batch_size = batch_size

    def scan(self) -> ScanResult:
        adj = extract_activation_graph(self.model, self.probe_dataset, self.num_samples, self.batch_size)
        fi = compute_fi(adj)
        lm = classify_layers(self.model, adj)
        from sal.fi import _infer_num_heads
        nh = _infer_num_heads(self.model)
        nl = adj.shape[0] // nh
        return ScanResult(fi_score=fi, layer_map=lm, adjacency=adj, num_layers=nl, num_heads_per_layer=nh)

class FIMonitor:
    def __init__(self, probe_dataset, interval=500, num_samples=200):
        self.probe_dataset = probe_dataset; self.interval = interval
        self.num_samples = num_samples; self._history = []; self._model = None

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._model = model

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.interval != 0 or not self._model: return
        try:
            adj = extract_activation_graph(self._model, self.probe_dataset, self.num_samples)
            fi = compute_fi(adj)
            self._history.append({"step": state.global_step, "fi_score": fi})
        except Exception as e:
            logger.warning(f"FI failed at step {state.global_step}: {e}")

    @property
    def history(self): return list(self._history)
