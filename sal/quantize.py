"""Quantization wrapper — one call over the backends people actually deploy.

This is deliberately **not** a quantization engine. It is a thin, honest wrapper
over bitsandbytes and ``torch.ao`` so the rest of sal-torch can ask for "INT4"
without every caller re-deriving which backend is installed, which one works on
this device, and which modules to leave alone.

    from sal import quantize, quantize_info

    print(quantize_info(model))
    small = quantize(model, method="int4")

Two rules it enforces, because getting either wrong quietly ruins a measurement:

* **The output head is never quantized.** ``lm_head`` / ``score`` /
  ``classifier`` stay in full precision, matching what every serious
  quantization pipeline does. Quantizing the vocabulary projection costs far
  more accuracy than it saves bytes.
* **GPT-2-style ``Conv1D`` is converted to ``nn.Linear`` first.** Every backend
  matches on ``nn.Linear``, so without this a GPT-2 model would report
  quantization results for a model in which nothing was quantized — same size,
  same accuracy, and no error to tell you.

Backends, in the order ``"auto"`` tries them:

``bitsandbytes``
    LLM.int8() for ``int8`` and NF4 for ``int4``, both CUDA-only. This is the
    real deployment path; install with ``pip install sal-torch[quant]``.
``torch_ao``
    Dynamic INT8 over ``nn.Linear``, CPU-only, no extra dependency. There is no
    INT4 here, so ``int4`` on this backend raises.
"""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

METHODS = ("int8", "int4")
BACKENDS = ("auto", "bitsandbytes", "torch_ao")

# Output projections stay in full precision — standard practice, and the thing
# that would otherwise dominate the accuracy loss.
DEFAULT_SKIP = ("lm_head", "score", "classifier", "pre_classifier", "embed_out", "head")


class QuantizationError(Exception):
    """Raised when no backend can serve the requested method."""


def _skipped(name: str, skip) -> bool:
    """Is this dotted module path excluded from quantization?

    Matches on whole path components, not substrings: ``head`` must skip the
    module actually called ``head`` without also catching ``head_proj``, and
    ``classifier`` must catch ``classifier.dense``.
    """
    return bool(set(name.split(".")) & set(skip))


# ------------------------------------------------------------------ availability
def has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — a broken install is the same as no install
        return False


def available_backends(method: str = "int8") -> list:
    """Backends that can serve ``method`` on this machine, best first."""
    out = []
    if has_bitsandbytes() and torch.cuda.is_available():
        out.append("bitsandbytes")
    if method == "int8":
        out.append("torch_ao")     # CPU dynamic INT8 is always available with torch
    return out


# ------------------------------------------------------------------ module swaps
def _conv1d_to_linear(model, skip):
    """Rewrite GPT-2 ``Conv1D`` (weight ``[in, out]``) as an equivalent Linear."""
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        return model

    def walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Conv1D):
                if _skipped(path, skip):
                    continue
                w = child.weight.data
                lin = nn.Linear(w.shape[0], w.shape[1], bias=child.bias is not None)
                lin.weight = nn.Parameter(w.t().contiguous(), requires_grad=False)
                if child.bias is not None:
                    lin.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
                setattr(parent, name, lin.to(w.device, w.dtype))
            else:
                walk(child, path)

    walk(model, "")
    return model


def _swap_linears(model, make_new, skip) -> int:
    """Replace every eligible ``nn.Linear`` via ``make_new``. Returns the count."""
    n = 0

    def walk(parent, prefix):
        nonlocal n
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if _skipped(path, skip):
                    continue
                new = make_new(child)
                if new is not None:
                    setattr(parent, name, new)
                    n += 1
            else:
                walk(child, path)

    walk(model, "")
    return n


def _bnb_int8(model):
    import bitsandbytes as bnb

    def make(old):
        new = bnb.nn.Linear8bitLt(old.in_features, old.out_features,
                                  bias=old.bias is not None, has_fp16_weights=False,
                                  threshold=6.0)
        new.weight = bnb.nn.Int8Params(old.weight.data.clone(), requires_grad=False,
                                       has_fp16_weights=False)
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.clone(), requires_grad=False)
        return new

    return make


def _bnb_int4(model, compute_dtype):
    import bitsandbytes as bnb

    def make(old):
        new = bnb.nn.Linear4bit(old.in_features, old.out_features,
                                bias=old.bias is not None,
                                compute_dtype=compute_dtype, quant_type="nf4")
        new.weight = bnb.nn.Params4bit(data=old.weight.data.clone(),
                                       requires_grad=False, quant_type="nf4")
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.clone(), requires_grad=False)
        return new

    return make


# --------------------------------------------------------------------- entrypoint
def quantize(model, method: str = "int8", backend: str = "auto", skip=DEFAULT_SKIP,
             compute_dtype=torch.float16, inplace: bool = False):
    """Quantize ``model``'s linear layers. Returns a new model unless ``inplace``.

    ``method`` is ``"int8"`` or ``"int4"``; ``backend`` is ``"auto"`` (try
    bitsandbytes, fall back to ``torch.ao``), ``"bitsandbytes"``, or ``"torch_ao"``.

    The chosen backend is recorded on the returned model as
    ``model._sal_quantization`` so downstream reports can state what actually ran
    rather than what was requested.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got '{method}'")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got '{backend}'")

    order = ([backend] if backend != "auto"
             else ["bitsandbytes", "torch_ao"])
    errors = []
    for name in order:
        try:
            out = _apply(model, method, name, skip, compute_dtype, inplace)
        except Exception as e:  # noqa: BLE001 — try the next backend, report if none work
            errors.append(f"{name}: {e}")
            if backend != "auto":
                raise QuantizationError(f"Backend '{name}' cannot do {method}: {e}") from e
            logger.info(f"Backend '{name}' unavailable for {method} ({e}); trying next.")
            continue
        return out

    raise QuantizationError(
        f"No backend available for {method}. Tried: {'; '.join(errors)}. "
        f"Install bitsandbytes for CUDA int4/int8 (pip install sal-torch[quant]); "
        f"torch.ao covers int8 on CPU only.")


def _apply(model, method: str, backend: str, skip, compute_dtype, inplace: bool):
    if backend == "bitsandbytes":
        if not has_bitsandbytes():
            raise RuntimeError("bitsandbytes is not installed")
        if not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes quantization needs CUDA")
        target = model if inplace else copy.deepcopy(model)
        _conv1d_to_linear(target, skip)
        make = _bnb_int8(target) if method == "int8" else _bnb_int4(target, compute_dtype)
        n = _swap_linears(target, make, skip)
        if n == 0:
            raise RuntimeError("no eligible nn.Linear layers found")
        target = target.cuda()
        backend_name = "bitsandbytes-int8" if method == "int8" else "bitsandbytes-nf4"

    elif backend == "torch_ao":
        if method != "int8":
            raise RuntimeError("torch.ao dynamic quantization has no int4 path")
        source = model if inplace else copy.deepcopy(model)
        target = source.cpu().float().eval()
        _conv1d_to_linear(target, skip)
        # Spec by module *name*, not by type: passing {nn.Linear} quantizes every
        # linear layer including the output head, silently ignoring `skip`.
        names = [n for n, m in target.named_modules()
                 if isinstance(m, nn.Linear) and not _skipped(n, skip)]
        if not names:
            raise RuntimeError("no eligible nn.Linear layers found")
        spec = {name: torch.ao.quantization.default_dynamic_qconfig for name in names}
        target = torch.ao.quantization.quantize_dynamic(target, spec, dtype=torch.qint8)
        n = len(names)
        backend_name = "torch-dynamic-int8"

    else:  # pragma: no cover - guarded by the caller
        raise RuntimeError(f"unknown backend '{backend}'")

    try:
        target._sal_quantization = {"method": method, "backend": backend_name,
                                    "layers_quantized": n}
    except Exception:  # noqa: BLE001 — some modules forbid attribute assignment
        pass
    logger.info(f"Quantized {n} linear layer(s) with {backend_name}.")
    return target


def model_size_mb(model) -> float:
    """Total tensor bytes in MB. Quantized parameters report packed storage."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total / 1e6


def quantize_info(model) -> dict:
    """What quantization would cost and save for this model, without running it.

    Sizes for the methods whose backends are unavailable are still reported —
    they are arithmetic, not measurements — but ``backends_available`` tells you
    which ones you can actually produce here.
    """
    original = model_size_mb(model)

    # Only linear weights get quantized; everything else keeps its dtype.
    skip = DEFAULT_SKIP
    quantizable_bytes = 0
    for name, mod in model.named_modules():
        is_linear = isinstance(mod, nn.Linear) or type(mod).__name__ == "Conv1D"
        if not is_linear or _skipped(name, skip):
            continue
        w = getattr(mod, "weight", None)
        if w is not None:
            quantizable_bytes += w.numel() * w.element_size()

    elem = next((p.element_size() for p in model.parameters()), 4)
    other = original - quantizable_bytes / 1e6
    return {
        "original_size_mb": round(original, 1),
        "int8_size_mb": round(other + (quantizable_bytes / elem) / 1e6, 1),
        "int4_size_mb": round(other + (quantizable_bytes / elem * 0.5) / 1e6, 1),
        "quantizable_mb": round(quantizable_bytes / 1e6, 1),
        "backends_available": available_backends("int8"),
        "int4_available": bool(available_backends("int4")),
    }
