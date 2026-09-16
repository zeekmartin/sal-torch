"""Evaluation metrics for self-supervised models.

A classification head gives you accuracy for free. A self-supervised encoder
does not — I-JEPA, MAE and DINO produce representations, and "did compression
hurt?" has to be answered about the representations themselves. These metrics
do that, and only the probe ones need labels at all:

  * :func:`linear_probe` — freeze the encoder, fit a linear classifier. The
    standard proxy for "how linearly separable is what this encoder knows".
  * :func:`knn_accuracy` — nearest-neighbour vote in feature space. No training,
    so it measures the geometry rather than what a head can be fitted to.
  * :func:`cka_similarity` — how close two models' representations are.
    Label-free, and the direct answer to "did compression change what the model
    computes".
  * :func:`representation_similarity` — mean cosine similarity, a cruder
    per-vector companion to CKA.
  * :func:`measure_latency`, :func:`count_params` — the other half of the
    trade: what the compression actually bought.

Everything here is torch + numpy. No timm, no datasets, no HF requirement.

Feature extraction is deliberately forgiving about model and batch shape (see
:func:`extract_features`) because the same call has to work for a timm ViT, an
HF ``IJepaModel`` and the tiny test transformer. Pass ``feature_fn=`` when the
guesswork gets it wrong.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "extract_features", "linear_probe", "knn_accuracy", "cka_similarity",
    "representation_similarity", "measure_latency", "count_params",
    "compression_report",
]


# --------------------------------------------------------------- batch plumbing
def _split_batch(batch):
    """Return ``(inputs, labels)`` for dict / tuple / bare-tensor batches."""
    if isinstance(batch, dict):
        labels = None
        for key in ("labels", "label", "targets", "y"):
            if key in batch:
                labels = batch[key]
                break
        inputs = {k: v for k, v in batch.items()
                  if k not in ("labels", "label", "targets", "y")}
        return inputs, labels
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return batch[0], None
        return batch[0], batch[1]
    return batch, None


def _model_device(model) -> torch.device:
    """The model's device, or CPU for a model that holds no parameters."""
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def _forward(model, inputs):
    """Run the encoder, preferring a dedicated feature path when one exists."""
    if isinstance(inputs, dict):
        return model(**inputs)
    fwd = getattr(model, "forward_features", None)   # timm convention
    if callable(fwd):
        return fwd(inputs)
    return model(inputs)


def _as_tensor(out) -> torch.Tensor:
    """Pull a tensor out of whatever the model returned."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("last_hidden_state", "pooler_output", "logits", "prediction_logits"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, dict):
        for key in ("last_hidden_state", "pooler_output", "logits", "features"):
            v = out.get(key)
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(
        f"Could not find a feature tensor in a {type(out).__name__} model output. "
        "Pass feature_fn=lambda model, inputs: <tensor> to say where it is.")


def _pool(t: torch.Tensor) -> torch.Tensor:
    """Reduce a model output to one vector per example."""
    if t.dim() == 2:                 # [B, D]
        return t
    if t.dim() == 3:                 # [B, tokens, D] — mean over tokens
        return t.mean(dim=1)
    if t.dim() == 4:                 # [B, C, H, W] — global average pool
        return t.mean(dim=(2, 3))
    return t.flatten(1)


def extract_features(model: nn.Module, dataloader, num_batches: Optional[int] = None,
                     device=None, feature_fn: Optional[Callable] = None):
    """Encode a dataloader into a feature matrix.

    Returns ``(features, labels)`` where ``features`` is a CPU float32 tensor of
    shape ``[N, D]`` and ``labels`` is a CPU int64 tensor of shape ``[N]``, or
    ``None`` when the batches carry no labels.

    The encoder is run under ``torch.no_grad()`` in eval mode; the model's
    original training flag is restored before returning. Outputs with a token or
    spatial axis are mean-pooled to one vector per example.

    ``feature_fn(model, inputs) -> Tensor`` overrides the forward pass entirely —
    use it for models whose features do not come out of ``forward()`` or
    ``forward_features()``.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    feats, labs = [], []
    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                inputs, y = _split_batch(batch)
                inputs = _to_device(inputs, device)
                out = feature_fn(model, inputs) if feature_fn else _forward(model, inputs)
                feats.append(_pool(_as_tensor(out)).float().cpu())
                if y is not None:
                    labs.append(y.detach().cpu())
    finally:
        model.train(was_training)

    if not feats:
        raise ValueError("Dataloader yielded no batches — nothing to extract.")
    features = torch.cat(feats, dim=0)
    labels = torch.cat(labs, dim=0).long() if len(labs) == len(feats) else None
    return features, labels


def _require_labels(labels, what: str):
    if labels is None:
        raise ValueError(
            f"{what} needs labels. Batches must be (inputs, labels) tuples or "
            "dicts with a 'labels' key.")


# ------------------------------------------------------------------- the probes
def linear_probe(model: nn.Module, train_loader, val_loader,
                 num_classes: Optional[int] = None, epochs: int = 100,
                 lr: float = 1e-2, batch_size: int = 256, device=None,
                 feature_fn: Optional[Callable] = None, seed: int = 0) -> float:
    """Train a linear classifier on frozen features; return validation accuracy.

    The encoder is frozen, so its features never change during probing — they
    are extracted once and the linear head is fitted on the cached matrix. That
    is mathematically the same as re-running the encoder each epoch and many
    times faster.

    ``num_classes`` is inferred from the labels when not given. Returns accuracy
    in [0, 1].

    The defaults are the standard probe recipe — 100 epochs of AdamW at 1e-2
    with a cosine decay. Because the features are cached, an "epoch" is a pass
    over a matrix in memory, so 100 of them cost less than a single pass through
    the encoder. Fewer epochs at a large batch size gives so few optimizer steps
    that the probe under-reports even perfectly separable features.
    """
    xtr, ytr = extract_features(model, train_loader, device=device, feature_fn=feature_fn)
    xva, yva = extract_features(model, val_loader, device=device, feature_fn=feature_fn)
    _require_labels(ytr, "linear_probe")
    _require_labels(yva, "linear_probe")

    if num_classes is None:
        num_classes = int(max(ytr.max().item(), yva.max().item())) + 1

    # Standardize: a linear head on raw ViT features converges much more slowly.
    mu, sigma = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-6)
    xtr, xva = (xtr - mu) / sigma, (xva - mu) / sigma

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    head = nn.Linear(xtr.shape[1], num_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    n = xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            loss = nn.functional.cross_entropy(head(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()
        sched.step()

    with torch.no_grad():
        pred = head(xva).argmax(dim=1)
    return float((pred == yva).float().mean().item())


def knn_accuracy(model: nn.Module, train_loader, val_loader, k: int = 20,
                 device=None, feature_fn: Optional[Callable] = None,
                 temperature: float = 0.07) -> float:
    """Weighted k-nearest-neighbour accuracy on L2-normalized features.

    No training at all: encode the train split, encode the val split, and let
    each val example be voted on by its ``k`` nearest train neighbours, weighted
    by ``exp(cosine / temperature)`` (the DINO/I-JEPA convention). Returns
    accuracy in [0, 1].
    """
    xtr, ytr = extract_features(model, train_loader, device=device, feature_fn=feature_fn)
    xva, yva = extract_features(model, val_loader, device=device, feature_fn=feature_fn)
    _require_labels(ytr, "knn_accuracy")
    _require_labels(yva, "knn_accuracy")

    xtr = nn.functional.normalize(xtr, dim=1)
    xva = nn.functional.normalize(xva, dim=1)
    k = max(1, min(k, xtr.shape[0]))
    num_classes = int(max(ytr.max().item(), yva.max().item())) + 1

    correct = 0
    for s in range(0, xva.shape[0], 256):           # chunked: the sim matrix is N*M
        chunk = xva[s:s + 256]
        sim = chunk @ xtr.T                          # cosine, both are unit norm
        top_sim, top_idx = sim.topk(k, dim=1)
        weights = (top_sim / temperature).exp()
        votes = torch.zeros(chunk.shape[0], num_classes)
        votes.scatter_add_(1, ytr[top_idx], weights)
        correct += int((votes.argmax(dim=1) == yva[s:s + 256]).sum().item())
    return correct / xva.shape[0]


# ------------------------------------------------------------------ similarity
def cka_similarity(model_a: nn.Module, model_b: nn.Module, dataloader,
                   num_batches: int = 50, device=None,
                   feature_fn: Optional[Callable] = None) -> float:
    """Linear CKA between two models' representations of the same data.

    1.0 means the two models compute the same representation up to an invertible
    linear map; above ~0.9 after compression is the usual "representations
    preserved" bar. Label-free, and invariant to feature dimension — so an
    original and a head-sliced model can be compared directly.
    """
    from sal.plasticity import _linear_cka

    fa, _ = extract_features(model_a, dataloader, num_batches, device, feature_fn)
    fb, _ = extract_features(model_b, dataloader, num_batches, device, feature_fn)
    if fa.shape[0] != fb.shape[0]:
        raise ValueError(
            f"Models saw a different number of examples ({fa.shape[0]} vs "
            f"{fb.shape[0]}). Use a non-shuffled dataloader.")
    return float(_linear_cka(fa.numpy().astype(np.float64),
                             fb.numpy().astype(np.float64)))


def representation_similarity(model_a: nn.Module, model_b: nn.Module, dataloader,
                              num_batches: int = 50, device=None,
                              feature_fn: Optional[Callable] = None) -> float:
    """Mean cosine similarity between the two models' feature vectors.

    Unlike CKA this is *not* invariant to rotation or rescaling of the feature
    space, and it needs both models to produce the same dimension. It is the
    stricter, more literal question: are these the same vectors?
    """
    fa, _ = extract_features(model_a, dataloader, num_batches, device, feature_fn)
    fb, _ = extract_features(model_b, dataloader, num_batches, device, feature_fn)
    if fa.shape != fb.shape:
        raise ValueError(
            f"Feature shapes differ ({tuple(fa.shape)} vs {tuple(fb.shape)}). "
            "Cosine similarity needs matching dimensions — use cka_similarity() "
            "to compare models of different widths.")
    return float(nn.functional.cosine_similarity(fa, fb, dim=1).mean().item())


# ------------------------------------------------------------------ the budget
def measure_latency(model: nn.Module, input_shape=(1, 3, 224, 224), device="cuda",
                    warmup: int = 10, runs: int = 100,
                    input_fn: Optional[Callable] = None) -> float:
    """Median single-forward latency in milliseconds.

    Median, not mean: a thermal blip or a stray kernel launch skews a mean and
    tells you nothing about the steady state you would ship.

    The model is moved to ``device`` for the measurement and moved back
    afterwards. ``input_fn(device) -> inputs`` supplies non-image inputs (a dict
    of token ids, say); otherwise a random tensor of ``input_shape`` is used.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "measure_latency(device='cuda') but no CUDA device is available. "
            "Pass device='cpu'.")

    original_device = _model_device(model)
    was_training = model.training
    model.eval().to(device)
    try:
        inputs = input_fn(device) if input_fn else torch.randn(*input_shape, device=device)
        sync = torch.cuda.synchronize if device == "cuda" else lambda: None

        with torch.no_grad():
            for _ in range(warmup):
                _forward(model, inputs)
            sync()
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _forward(model, inputs)
                sync()
                times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        model.to(original_device)
        model.train(was_training)
    return float(np.median(times))


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    """Total parameter count, optionally restricted to those requiring grad."""
    ps = model.parameters()
    if only_trainable:
        ps = (p for p in ps if p.requires_grad)
    return int(sum(p.numel() for p in ps))


# ----------------------------------------------------------------- the report
def compression_report(original_model: nn.Module, compressed_model: nn.Module,
                       dataloader, name: str = "", train_loader=None,
                       val_loader=None, num_batches: int = 50, device=None,
                       feature_fn: Optional[Callable] = None,
                       probe_dataset=None, k: int = 20,
                       input_shape=(1, 3, 224, 224),
                       input_fn: Optional[Callable] = None) -> dict:
    """One dict answering "what did this compression cost, and what did it buy?".

    Always computed: parameter counts, compression ratio, CKA against the
    original, and CPU latency for both models (plus GPU latency when CUDA is
    present).

    Computed when the inputs are there: ``linear_probe`` and ``knn_accuracy``
    (need ``train_loader`` **and** ``val_loader`` with labels), and the Fragility
    Index before/after (needs ``probe_dataset``).

    Optional metrics that cannot be computed are reported as ``None`` with the
    reason under ``"skipped"`` — an unexplained ``None`` in a benchmark table is
    indistinguishable from a zero.

    ``input_shape`` / ``input_fn`` are handed to :func:`measure_latency`; the
    default is one 224x224 RGB image, so override them for anything that is not
    a vision backbone. A model that will not accept the timing input has its
    latency reported as ``None`` rather than failing the whole report — the
    quality metrics are the expensive part and are already computed by then.
    """
    skipped: dict[str, str] = {}
    orig_n = count_params(original_model)
    comp_n = count_params(compressed_model)

    report = {
        "name": name,
        "original_params": orig_n,
        "compressed_params": comp_n,
        "compression_ratio": (orig_n / comp_n) if comp_n else float("nan"),
        "params_removed_pct": 100.0 * (1 - comp_n / orig_n) if orig_n else float("nan"),
        "cka_similarity": cka_similarity(original_model, compressed_model, dataloader,
                                         num_batches, device, feature_fn),
        "linear_probe": None,
        "knn_accuracy": None,
        "latency_cpu_ms": None,
        "latency_cpu_ms_original": None,
        "latency_gpu_ms": None,
        "latency_gpu_ms_original": None,
        "fi_before": None,
        "fi_after": None,
    }

    if train_loader is not None and val_loader is not None:
        report["linear_probe"] = linear_probe(compressed_model, train_loader, val_loader,
                                              device=device, feature_fn=feature_fn)
        report["knn_accuracy"] = knn_accuracy(compressed_model, train_loader, val_loader,
                                              k=k, device=device, feature_fn=feature_fn)
    else:
        skipped["linear_probe"] = "needs train_loader and val_loader with labels"
        skipped["knn_accuracy"] = "needs train_loader and val_loader with labels"

    def timed(model, device, runs):
        try:
            return measure_latency(model, input_shape=input_shape, device=device,
                                   runs=runs, input_fn=input_fn)
        except Exception as e:
            skipped.setdefault(f"latency_{device}_ms", f"{type(e).__name__}: {e}")
            logger.warning("Latency on %s skipped in compression_report: %s", device, e)
            return None

    report["latency_cpu_ms"] = timed(compressed_model, "cpu", 20)
    report["latency_cpu_ms_original"] = timed(original_model, "cpu", 20)

    if torch.cuda.is_available():
        report["latency_gpu_ms"] = timed(compressed_model, "cuda", 100)
        report["latency_gpu_ms_original"] = timed(original_model, "cuda", 100)
    else:
        skipped["latency_gpu_ms"] = "no CUDA device available"

    if probe_dataset is not None:
        from sal.scanner import FIScanner
        try:
            report["fi_before"] = FIScanner(original_model, probe_dataset).scan().fi_score
            report["fi_after"] = FIScanner(compressed_model, probe_dataset).scan().fi_score
        except Exception as e:                      # arch not introspectable, etc.
            skipped["fi"] = f"{type(e).__name__}: {e}"
            logger.warning("FI scan skipped in compression_report: %s", e)
    else:
        skipped["fi"] = "needs probe_dataset"

    if skipped:
        report["skipped"] = skipped
    return report
