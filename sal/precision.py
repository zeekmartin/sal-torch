"""What precision is this model actually in, and would a load narrow it?

The bug class, measured. OPT's config carries ``torch_dtype: float16``, so
``from_config`` builds half-precision parameters and ``load_state_dict``
narrows an fp32 checkpoint **without complaining**. The first zero-shot pass of
the 6.7B campaign ran that way and had to be re-run in float32: nothing raised,
nothing warned, and the numbers were simply wrong. Public data archive, DOI
10.5281/zenodo.24391200, ``tables/REVIEW_FACTS.md`` section A5, jobs 23739252
and 23742387.

Two different checks come out of that:

* **a load**, where a checkpoint is about to go into parameters of a different
  dtype. If the parameters are narrower, the load loses precision and
  ``check_load_precision`` raises.
* **a model already in hand**, where there is no checkpoint to compare and the
  question is what the numbers about to be measured will mean. Half-precision
  parameters, or a config whose declared dtype is narrower than the parameters
  it describes, are recorded and warned about. The campaign's own figures were
  produced with fp32 weights under autocast, so a figure measured on fp16
  weights is not comparable with them.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class PrecisionError(Exception):
    """Raised when a load would silently lose precision."""


def _width(dtype) -> tuple:
    """``(bits, mantissa bits)``, which is what "narrower" has to mean.

    float32 to bfloat16 keeps the exponent range and loses 16 bits of mantissa;
    float16 to bfloat16 keeps the width and loses 3 bits of mantissa. Ordering
    on bits alone would call the second one lossless.
    """
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        return (0, 0)
    fi = torch.finfo(dtype)
    return (fi.bits, _mantissa(dtype))


def _mantissa(dtype) -> int:
    return {torch.float64: 52, torch.float32: 23, torch.bfloat16: 7,
            torch.float16: 10}.get(dtype, 0)


def _as_dtype(value) -> Optional[torch.dtype]:
    """A config's ``torch_dtype`` may be a dtype, a string, or absent."""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return getattr(torch, value.replace("torch.", ""), None)
    return None


def _dominant(dtypes) -> Optional[torch.dtype]:
    floats = [d for d in dtypes if isinstance(d, torch.dtype)
              and d.is_floating_point]
    if not floats:
        return None
    counts: dict = {}
    for d in floats:
        counts[d] = counts.get(d, 0) + 1
    # The narrowest wins a tie: it is the one that bounds what the model can
    # represent, and the point of this module is not to flatter the model.
    return sorted(counts, key=lambda d: (-counts[d], _width(d)))[0]


def narrower(target, source) -> bool:
    """True when moving ``source`` into ``target`` loses precision."""
    if target is None or source is None or target == source:
        return False
    tb, tm = _width(target)
    sb, sm = _width(source)
    return tb < sb or tm < sm


def model_precision(model) -> dict:
    """Parameter dtype, config dtype, and whether the two disagree."""
    params = _dominant([p.dtype for p in model.parameters()])
    buffers = _dominant([b.dtype for b in model.buffers()])
    config = _as_dtype(getattr(getattr(model, "config", None), "torch_dtype",
                               None))
    mixed = sorted({str(p.dtype) for p in model.parameters()
                    if p.dtype.is_floating_point})
    return {
        "param_dtype": str(params) if params else None,
        "buffer_dtype": str(buffers) if buffers else None,
        "config_dtype": str(config) if config else None,
        "mixed_param_dtypes": mixed if len(mixed) > 1 else None,
        "half_precision_weights": bool(params and _width(params)[0] <= 16),
        "config_narrower_than_params": narrower(config, params),
    }


def check_load_precision(model, state_dict=None, *, strict: bool = True) -> dict:
    """Would loading ``state_dict`` into ``model`` lose precision?

    Returns the dtypes it found plus ``narrowing``. With a ``state_dict``,
    ``narrowing`` is true when the parameters are narrower than the checkpoint,
    which is the silent OPT case; without one it is true when the config
    declares a dtype narrower than the parameters, which is the same failure
    one save-and-reload later. Under ``strict=True`` a narrowing load raises
    :class:`PrecisionError`; otherwise it warns.
    """
    info = model_precision(model)
    ckpt = None
    if state_dict is not None:
        ckpt = _dominant([v.dtype for v in state_dict.values()
                          if isinstance(v, torch.Tensor)])
    info["checkpoint_dtype"] = str(ckpt) if ckpt else None

    params = _dominant([p.dtype for p in model.parameters()])
    if ckpt is not None:
        info["narrowing"] = narrower(params, ckpt)
        detail = (f"the checkpoint is {ckpt} and the parameters it is being "
                  f"loaded into are {params}")
    else:
        info["narrowing"] = info["config_narrower_than_params"]
        detail = (f"the config declares {info['config_dtype']} and the "
                  f"parameters are {info['param_dtype']}, so a save and "
                  "reload through that config would narrow them")

    if info["narrowing"]:
        message = (
            f"this load loses precision: {detail}. load_state_dict does not "
            "complain about it. The 6.7B zero-shot pass of this project's own "
            "campaign ran that way and had to be re-run in float32, because "
            "OPT's config carries torch_dtype float16 and from_config builds "
            "half-precision parameters. Load in float32 and let autocast "
            "handle the arithmetic, or pass strict=False if the narrowing is "
            "what you meant.")
        if strict:
            raise PrecisionError(message)
        logger.warning(message)
    elif info["half_precision_weights"]:
        logger.warning(
            "this model's weights are %s. Every figure this project publishes "
            "was measured with fp32 weights under autocast (bfloat16 for "
            "training and recovery, float16 for scoring), so a number measured "
            "here is not comparable with them.", info["param_dtype"])
    if info["mixed_param_dtypes"]:
        logger.warning("this model's parameters are in more than one floating "
                       "dtype: %s", ", ".join(info["mixed_param_dtypes"]))
    return info
