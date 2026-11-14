"""High-level structural analysis: FIScanner and FIMonitor."""
from __future__ import annotations
import warnings
import json, logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sal.fi import LayerClass, classify_layers, compute_fi, extract_activation_graph

logger = logging.getLogger(__name__)

def _check_probe(probe_data, strict: bool = False) -> None:
    """Warn (or raise) when a probe is not what FI assumes it is.

    FI is a property of the **(model, probe)** pair, and a malformed probe
    does not fail -- it returns a different number that looks entirely
    reasonable. The two cases worth catching:

    * **An integer probe.** ``nn.Conv2d`` promotes an integer tensor
      silently, so a scan over the raw ``uint8`` images runs on inputs two
      orders of magnitude outside a vision transformer's expected range and
      reports a plausible score.
    * **A float probe that was never normalized.** Values in ``[0, 1]`` or
      ``[0, 255]`` are the same mistake one step later. A mean/std-normalized
      image tensor straddles zero; neither of those does.

    The default is a warning rather than an exception, because this check
    cannot see custom preprocessing and because a scanner that refuses to run
    is worse than one that says what it suspects. Pass
    ``strict_probe=True`` to make it an error.
    """
    if not probe_data:
        return
    for key in ("pixel_values", "input_values"):
        first = None
        for batch in probe_data:
            if isinstance(batch, dict) and key in batch:
                first = batch[key]
                break
        if first is None:
            continue
        problem = None
        if not torch.is_floating_point(first):
            problem = (
                "the probe's {!r} has dtype {} -- an integer tensor. This is "
                "almost always the un-preprocessed image cache: layers "
                "promote it silently, so the scan completes and the score "
                "means something else.".format(key, first.dtype))
        else:
            lo = float(first.min())
            if lo >= -0.01:
                problem = (
                    "the probe's {!r} spans [{:.3f}, {:.3f}] and never goes "
                    "negative, so it looks un-normalized. A mean/std-"
                    "normalized tensor straddles zero; raw values sit in "
                    "[0, 1] or [0, 255].".format(
                        key, lo, float(first.max())))
        if problem is None:
            return
        msg = (problem + " The score is a property of the (model, probe) "
               "pair, so this changes the result rather than breaking it. "
               "Preprocess the probe the way the model was trained, or pass "
               "strict_probe=False to silence this.")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return


@dataclass
class ScanResult:
    """One FI scan: the score, the layer labels, and the probe it came from.

    ``num_samples`` and ``batch_size`` are part of the result, not of the call
    that produced it. FI is a property of the **(model, probe) pair**, and the
    probe's *size* moves it as much as its contents do: the same ViT-B/16
    checkpoint, the same normalised probe construction and the same edge
    density read **FI 0.151 at 64 images and 0.071-0.082 at 500**
    (``results/lumi_v060/EXP42_FI_THRESHOLDS.md``). That is a factor of two,
    larger than any compression effect in that campaign.

    So an FI figure without its sample count is not interpretable, exactly as
    a latency figure without its batch size is not, and two FI scores may only
    be compared when they were measured at the same ``num_samples``. The field
    is carried here, printed by :attr:`summary`, and written by :meth:`save`
    so the pairing survives into whatever reads the file.
    """

    fi_score: float
    layer_map: dict[int, LayerClass]
    adjacency: np.ndarray
    num_layers: int = 0
    num_heads_per_layer: int = 0
    num_samples: int = 0
    batch_size: int = 0

    @property
    def edge_count(self) -> int:
        """Edges in the graph FI was computed on."""
        a = (np.asarray(self.adjacency) != 0)
        return int(a.sum() // 2) - int(np.trace(a) // 2)

    @property
    def fi_resolution(self) -> float:
        """The smallest FI change this graph can express: one edge.

        Worth comparing a delta against, but it is **not** the floor. On
        ViT-B/16 the probe-to-probe spread of FI on an unchanged model was
        measured at eleven times this (EXP42_FI_THRESHOLDS.md), so what limits
        a comparison is which images were drawn, not what the graph can
        express.
        """
        e = self.edge_count
        return 1.0 / e if e else 0.0

    @property
    def critical_layers(self): return [i for i,c in self.layer_map.items() if c == LayerClass.CRITICAL]
    @property
    def immune_layers(self): return [i for i,c in self.layer_map.items() if c == LayerClass.IMMUNE]
    @property
    def buffer_layers(self): return [i for i,c in self.layer_map.items() if c == LayerClass.BUFFER]
    @property
    def summary(self):
        # The probe size rides along with the score on purpose: see the class
        # docstring. A bare "FI=0.0796" is not a quotable number.
        probe = (f" | probe {self.num_samples} samples" if self.num_samples
                 else " | probe size NOT RECORDED")
        return (f"FI={self.fi_score:.4f}{probe} | {len(self.immune_layers)} "
                f"immune, {len(self.buffer_layers)} buffer, "
                f"{len(self.critical_layers)} critical")

    def save(self, path: str):
        p = Path(path)
        if p.suffix == ".json":
            data = {"fi_score": self.fi_score, "num_layers": self.num_layers,
                    "num_heads_per_layer": self.num_heads_per_layer,
                    # Without these the score cannot be compared with another
                    # one later, which is the whole point of saving it.
                    "num_samples": self.num_samples,
                    "batch_size": self.batch_size,
                    "graph_edges": self.edge_count,
                    "layer_classification": {str(k): v.value for k,v in self.layer_map.items()}}
            p.write_text(json.dumps(data, indent=2))
        elif p.suffix == ".pdf":
            from sal.visualize import render_fi_pdf
            render_fi_pdf(self, str(p))
        else:
            raise ValueError(f"Unsupported: {p.suffix}. Use .json or .pdf")

class FIScanner:
    def __init__(self, model, probe_dataset, num_samples=500, batch_size=16,
                 strict_probe=False):
        self.model = model; self.probe_dataset = probe_dataset
        self.num_samples = num_samples; self.batch_size = batch_size
        self.strict_probe = strict_probe
        _check_probe(probe_dataset, strict_probe)

    def scan(self) -> ScanResult:
        adj = extract_activation_graph(self.model, self.probe_dataset, self.num_samples, self.batch_size)
        fi = compute_fi(adj)
        lm = classify_layers(self.model, adj)
        from sal.fi import _infer_num_heads
        nh = _infer_num_heads(self.model)
        nl = adj.shape[0] // nh
        return ScanResult(fi_score=fi, layer_map=lm, adjacency=adj, num_layers=nl,
                          num_heads_per_layer=nh, num_samples=self.num_samples,
                          batch_size=self.batch_size)

class FIMonitor:
    def __init__(self, probe_dataset, interval=500, num_samples=200,
                 strict_probe=False):
        _check_probe(probe_dataset, strict_probe)
        self.probe_dataset = probe_dataset; self.interval = interval
        self.num_samples = num_samples; self._history = []; self._model = None

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._model = model

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.interval != 0 or not self._model: return
        try:
            adj = extract_activation_graph(self._model, self.probe_dataset, self.num_samples)
            fi = compute_fi(adj)
            # num_samples on every row: a history is a sequence of comparisons,
            # and they are only valid at a fixed probe size.
            self._history.append({"step": state.global_step, "fi_score": fi,
                                  "num_samples": self.num_samples})
        except Exception as e:
            logger.warning(f"FI failed at step {state.global_step}: {e}")

    @property
    def history(self): return list(self._history)
