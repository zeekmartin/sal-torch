"""Repair a compressed model, and show what the repair did.

Recovery is where the accuracy in post-hoc pruning comes from. At 50% of
attention heads a ViT-B/16 on ImageNet-1k reads 9.56% one-shot and 77.642%
after two epochs against a dense control trained for the same two epochs, so a
pipeline that prunes and scores without repairing reports a number 68 points
below what the method delivers.

Three things in this module are not conveniences:

* **The gradient norm is clipped by default, and logged every step.** The
  research harness that produced the decoder cells ran an unclipped AdamW under
  a one-cycle schedule whose learning rate was still climbing through the first
  tenth of the run, and under it recovery diverged on the most damaged arm on
  every seed. The result files carried only the first and last loss, and the
  first and last of a curve that rises in the middle and comes most of the way
  back look unremarkable, so the diagnosis had to be reconstructed from job
  logs. A stage that cannot show its own instability is not reporting.

* **The warmup does not ramp the learning rate while the model is at its most
  damaged.** ``schedule="warmup_cosine"`` with ``warmup_ratio=0.05`` is the
  measured vision recipe, not a new guess.

* **A mask is enforced, and a violation raises.** An ordinary fine-tune fills
  the zeros of an unstructured or 2:4 model back in, and the model stops being
  sparse while still carrying the label. Measured drift was exactly 0.0 on
  24 of 24 recovery runs, so the invariant is achievable and a violation means
  something is wrong.

``schedule="one_cycle"`` and ``max_grad_norm=None`` exist so that a published
cell can be reproduced through this code exactly as it was measured. They are
not recommended settings.
"""
from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

#: Parameter-name prefixes treated as the freshly initialised output head, which
#: the measured recipe trains at a higher learning rate than the backbone.
HEAD_PREFIXES = ("classifier", "head", "lm_head", "score", "fc")

SCHEDULES = ("warmup_cosine", "one_cycle")

#: Accepted values of ``domain`` that mean "not the target domain". Spelled out
#: rather than implied by a default, because the choice of recovery corpus is
#: the difference between repairing damage and measuring a domain shift.
GENERIC_DOMAIN = "generic"


class _Unset:
    """Tells "the caller said 2" apart from "nobody said anything"."""

    def __repr__(self):
        return "<unset>"


UNSET = _Unset()

#: exp29 measured the curve and it peaks here: 5 epochs is below 2 on 3/3
#: seeds in FP and INT4, and two epochs is 39% of the cost of five. It is a
#: measured default and not an optimum, and the dense control's own curve peaks
#: at 1, so the two do not share a knee.
DEFAULT_EPOCHS = 2

#: Measured on OPT-1.3B: 32M tokens of C4 make the UNPRUNED model worse on raw
#: WikiText-2 by 0.6398 perplexity (pretrained 14.6240 against a dense control
#: of 15.3260 / 15.3536 / 15.1120 over three seeds). At 6.7B the same corpus
#: costs the dense control +1.0748 while 50% unstructured pruning costs +1.1987
#: one-shot, so the recovered arm ends worse than its own one-shot model on 3 of
#: 3 seeds. Public data archive, DOI 10.5281/zenodo.24391200.
GENERIC_DOMAIN_WARNING = (
    "recover(domain='generic'): the recovery corpus is not the domain the model "
    "is evaluated on. Measured consequence: on OPT-1.3B, 32M tokens of C4 made "
    "the UNPRUNED model worse on raw WikiText-2 by 0.6398 perplexity, and at "
    "6.7B the same corpus cost the dense control more than 50% unstructured "
    "pruning cost it one-shot, so the recovered arm ended worse than its own "
    "one-shot model on 3 of 3 seeds. Pass a control to have this checked for "
    "your run rather than assumed.")


class RecoveryError(Exception):
    """Raised when a recovery run cannot be reported honestly."""


@dataclass
class RecoveryControl:
    """The dense model trained for the SAME budget, with nothing pruned.

    A cost is only a pruning cost when it is taken against this. A delta
    against the pretrained checkpoint is the sum of two effects, and reporting
    it as a cost is how a model that pruning damaged gets reported as improved.
    """
    accuracy: float
    epochs: Optional[int] = None
    tokens: Optional[int] = None
    seeds: int = 1
    metric: str = "accuracy"

    @property
    def budget(self) -> tuple:
        return ("epochs", self.epochs) if self.tokens is None else ("tokens", self.tokens)

    def matches(self, epochs: Optional[int], tokens: Optional[int]) -> bool:
        """True only when the control was trained on the arm's own budget."""
        if self.tokens is not None or tokens is not None:
            return self.tokens is not None and tokens is not None and self.tokens == tokens
        return self.epochs is not None and epochs is not None and self.epochs == epochs


# ------------------------------------------------------------------ statistics
def resolve_budget(epochs, tokens) -> tuple:
    """``(epochs, tokens)`` with exactly one of them set.

    Unset with no token budget is the measured two epochs. Unset with one is
    token mode. Both given is refused, because then nothing says which one the
    schedule was built from, and the schedule is the whole difference between
    a two-epoch recovery and the first two epochs of a longer one.
    """
    if tokens is not None:
        if epochs is not UNSET and epochs is not None:
            raise ValueError(
                "recover() takes exactly one budget: epochs= for vision "
                "models and encoders, tokens= for decoders. Both were given "
                "(epochs=%r, tokens=%r), which leaves it undefined which one "
                "the schedule was built from." % (epochs, tokens))
        return None, tokens
    if epochs is UNSET:
        return DEFAULT_EPOCHS, None
    if epochs is None:
        raise ValueError(
            "recover() takes exactly one budget and got neither: pass epochs= "
            "for a vision model or an encoder, or tokens= for a decoder.")
    return epochs, None


def max_rise_above_running_min(curve) -> tuple:
    """The largest rise above the running minimum, and the step it peaked at.

    This is the statistic that separates a run that blew up and came back from
    one that merely tracked a hard batch: the endpoints of both look alike.
    Returns ``(rise, step)``, with ``(0.0, None)`` for a curve that never rises.
    """
    best_rise, best_step, running = 0.0, None, None
    for i, v in enumerate(curve):
        v = float(v)
        if running is None or v < running:
            running = v
        rise = v - running
        if rise > best_rise:
            best_rise, best_step = rise, i
    return best_rise, best_step


# ----------------------------------------------------------------------- masks
def capture_sparsity_masks(model, params=None) -> dict:
    """``{parameter name: bool tensor}`` of the currently non-zero weights.

    Only 2-D weights with a zero anywhere are captured: a dense model has no
    mask, which is why ``preserve_sparsity=True`` is a no-op on a structurally
    sliced model rather than an error.
    """
    masks = {}
    for name, p in model.named_parameters():
        if params is not None and name not in params:
            continue
        if p.dim() < 2:
            continue
        nz = p.data != 0
        if bool(nz.all()):
            continue
        masks[name] = nz.clone()
    return masks


def _mask_stats(model, masks: dict) -> dict:
    """Zero fractions over the masked parameters, and the escaped-zero count."""
    total = zeros = escaped = 0
    for name, p in model.named_parameters():
        m = masks.get(name)
        if m is None:
            continue
        total += m.numel()
        zeros += int((p.data == 0).sum())
        escaped += int((p.data[~m] != 0).sum())
    return {"n_masked_params": len(masks), "n_weights": total,
            "zero_fraction": (zeros / total) if total else 0.0,
            "escaped_zeros": escaped}


# ------------------------------------------------------------------- schedules
def _warmup_cosine(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(warmup_ratio * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return factor, warmup


#: The harness's OneCycleLR settings, which the published decoder cells ran:
#: start at ``max_lr / 25``, cosine up to ``max_lr`` over ``pct_start`` of the
#: run, cosine down to ``max_lr / 25 / 1e4``.
ONE_CYCLE_DIV_FACTOR = 25.0
ONE_CYCLE_FINAL_DIV_FACTOR = 1e4


def _cos_anneal(start: float, end: float, pct: float) -> float:
    """torch's ``OneCycleLR._annealing_cos``, including its lack of clamping."""
    return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)


def _one_cycle(total_steps: int, pct_start: float):
    """``OneCycleLR``'s schedule as a factor over ``max_lr``.

    Reproduced rather than delegated for one reason: ``OneCycleLR`` divides by
    zero whenever ``pct_start * total_steps`` is exactly 1, which a unit test
    on a tiny fixture hits immediately, and a schedule that cannot be exercised
    on the tiny fixture cannot be tested (cardinal rule 11). Everything else
    follows torch exactly, because **paper mode has to reproduce the schedule
    the archived numbers were measured under**: the up phase ends at the float
    ``pct_start * total_steps - 1``, the within-phase percentage is not
    clamped, and the anneal is torch's own. ``tests/test_recovery_schedules.py``
    steps the two side by side at the campaign's real step counts and requires
    agreement to 1e-12 relative.

    Returns ``(factor, peak_step)``.
    """
    low = 1.0 / ONE_CYCLE_DIV_FACTOR
    final = low / ONE_CYCLE_FINAL_DIV_FACTOR
    up_end = float(pct_start * total_steps) - 1.0
    down_end = float(total_steps - 1)

    def factor(step: int) -> float:
        step = float(step)
        if step <= up_end:
            # start_step is 0.0 in the up phase, so the span is up_end itself.
            pct = (step / up_end) if up_end else 1.0
            return _cos_anneal(low, 1.0, pct)
        span = down_end - up_end
        pct = ((step - up_end) / span) if span else 1.0
        return _cos_anneal(1.0, final, pct)

    return factor, max(0, int(up_end))


def _param_groups(model, lr: float, head_lr_multiplier: float) -> list:
    head, body = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.split(".")[0] in HEAD_PREFIXES else body).append(p)
    groups = [{"params": body, "lr": lr}]
    if head:
        groups.append({"params": head, "lr": lr * head_lr_multiplier})
    return groups


#: What the campaign's own runs used, and what a figure is comparable with.
ARCHIVED_AMP = "bfloat16 autocast on MI250X (scoring: float16 autocast)"


def bf16_supported(device_type: str) -> bool:
    """Does this device do bfloat16 autocast? Probed, never assumed."""
    if device_type == "cuda":
        try:
            return bool(torch.cuda.is_available()
                        and torch.cuda.is_bf16_supported())
        except (AssertionError, RuntimeError):      # no driver, no device
            return False
    return device_type == "cpu"     # torch has CPU bf16, see select_precision


def select_precision(device_type: str, requested=None, *, supports_bf16=None):
    """``(amp_dtype, needs_scaler, note)`` for this device and this request.

    ``requested`` is ``None`` for no autocast at all, ``"auto"`` to take the
    best this device has, or an explicit dtype, which is a **pin**:

    * a pin the device can honour is honoured;
    * a pin of bfloat16 on an accelerator without bfloat16 **raises**, because
      the only reason to pin it is to reproduce a published cell and a silent
      substitution would make the run something else;
    * a pin on a CPU falls back to fp32 with a note, since nobody reproduces a
      published accelerator figure on a CPU and CPU autocast is a third dtype
      policy rather than a cheaper version of the first two.

    ``supports_bf16`` overrides the probe, which is what makes every branch
    here testable on a machine with no accelerator at all.
    """
    probe = (bf16_supported(device_type) if supports_bf16 is None
             else bool(supports_bf16))
    if requested is None:
        return None, False, "fp32, no autocast"

    if isinstance(requested, str) and requested.lower() == "auto":
        if device_type == "cpu":
            return (None, False,
                    "fp32 on a CPU: CPU autocast is a different dtype policy "
                    "from the one the archived runs used (%s)" % ARCHIVED_AMP)
        if probe:
            return torch.bfloat16, False, "bfloat16 autocast, as the archive"
        return (torch.float16, True,
                "float16 autocast with a GradScaler: this device has no "
                "bfloat16, and the archived figures were measured under %s"
                % ARCHIVED_AMP)

    dtype = (getattr(torch, requested.replace("torch.", ""))
             if isinstance(requested, str) else requested)
    if device_type == "cpu":
        return (None, False,
                "fp32 on a CPU: %s was pinned, and CPU autocast is a "
                "different dtype policy from the archived %s"
                % (dtype, ARCHIVED_AMP))
    if dtype is torch.bfloat16 and not probe:
        raise RecoveryError(
            "bfloat16 autocast was pinned and %s does not support it. A pin "
            "is for reproducing a published cell, so it is refused rather "
            "than substituted: pass amp_dtype='auto' to take float16 with a "
            "GradScaler instead, and do not call the result a reproduction."
            % device_type)
    needs_scaler = dtype is torch.float16
    return (dtype, needs_scaler,
            "%s autocast%s, pinned" % (str(dtype).replace("torch.", ""),
                                       " with a GradScaler" if needs_scaler
                                       else ""))


def backward_and_clip(loss, params, max_grad_norm, optimizer, scaler=None):
    """One backward, one clip, one step, in the order that keeps the log true.

    With a scaler the gradients must be **unscaled before** they are clipped
    or measured, or the logged norm is the loss scale rather than the
    gradient. Returns the norm before clipping, which is what the result
    records.
    """
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        params, max_grad_norm if max_grad_norm else float("inf"))
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return float(norm)


def autocast_for(device_type: str, amp_dtype):
    """The autocast context a run should use, or a null one.

    Off on CPU whatever is asked for: the archived runs autocast on an
    accelerator, and CPU autocast is a different dtype policy, so honouring
    the flag there would quietly measure a third thing. Returned rather than
    entered inline so that the choice is testable without the accelerator.
    """
    if amp_dtype is None or device_type == "cpu":
        return contextlib.nullcontext()
    if isinstance(amp_dtype, str):
        amp_dtype = getattr(torch, amp_dtype.replace("torch.", ""))
    return torch.autocast(device_type, dtype=amp_dtype)


def _tokens_in(batch) -> int:
    if isinstance(batch, dict):
        for key in ("input_ids", "pixel_values", "inputs_embeds"):
            if key in batch:
                return int(batch[key].numel() if key == "input_ids"
                           else batch[key].shape[0])
        first = next(iter(batch.values()))
        return int(first.shape[0])
    return int(batch.shape[0])


# ------------------------------------------------------------------- the stage
def recover_model(model, train_dataset, *,
                  domain: str,
                  epochs=UNSET,
                  tokens: Optional[int] = None,
                  lr: float = 5e-5,
                  head_lr_multiplier: float = 10.0,
                  weight_decay: float = 0.05,
                  batch_size: int = 8,
                  max_grad_norm: Optional[float] = 1.0,
                  schedule: str = "warmup_cosine",
                  warmup_ratio: float = 0.05,
                  max_lr: Optional[float] = None,
                  pct_start: float = 0.10,
                  preserve_sparsity: bool = True,
                  masks: Optional[dict] = None,
                  enforce_masks: bool = True,
                  amp_dtype=None,
                  seed: int = 0,
                  device=None,
                  optimizer=None) -> dict:
    """Fine-tune a compressed model and return everything the run measured.

    ``domain`` is required and has no default: it names the domain the
    recovery data comes from and is recorded with the result. A corpus from
    outside the evaluation domain can cost more than the pruning it repairs,
    so there is no silent default. Pass ``"generic"`` to proceed knowingly,
    which warns with the measured reason.

    ``epochs`` and ``tokens`` are mutually exclusive: a decoder's budget is a
    token count, and epochs over a corpus shard is not a unit that transfers
    between scales.

    ``amp_dtype`` is ``None`` for no autocast, ``"auto"`` to take the best
    precision the device has, or an explicit dtype, which is a pin. Every
    recovery in the campaign ran under **bfloat16** autocast and every score
    under **float16** autocast, which are different and were both written up
    as "autocast bfloat16" until the pre-report audit, so ``"auto"`` prefers
    bfloat16 and falls back to float16 with a GradScaler where the device has
    no bfloat16, or to fp32 on a CPU. Whatever runs is warned about in one
    line and recorded in the result, because a run at a different precision
    from the archived one is not comparable with it. A pinned bfloat16 on an
    accelerator without bfloat16 raises rather than substituting: see
    :func:`select_precision`.

    ``masks`` may be supplied by the caller (``{parameter name: bool tensor}``
    of the weights to keep non-zero); if it is ``None`` and
    ``preserve_sparsity`` is set, the zero pattern of the model as handed in is
    captured and defended. ``enforce_masks=False`` runs without the gradient
    hook and still measures the drift, which is how the test that the invariant
    is actually enforced can fail honestly.
    """
    from sal.fi import _iter_data, _to_dev

    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(
            "recover_model() needs domain=: a short string naming the domain "
            "the recovery data comes from, which is recorded with the result. "
            "There is no default, because a recovery corpus that does not "
            "match the evaluation domain can cost more than the pruning it "
            "repairs. Pass domain='generic' to proceed knowingly.")
    if domain.strip().lower() == GENERIC_DOMAIN:
        logger.warning(GENERIC_DOMAIN_WARNING)
    epochs, tokens = resolve_budget(epochs, tokens)
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be at least 1")
    if tokens is not None and tokens < 1:
        raise ValueError("tokens must be at least 1")

    device = device or next(model.parameters()).device
    torch.manual_seed(seed)
    batches = list(_iter_data(train_dataset, batch_size))
    if not batches:
        raise ValueError("recover() got an empty training dataset")

    if tokens is None:
        total_steps = max(1, len(batches) * epochs)
    else:
        per_batch = max(1, _tokens_in(_to_dev(batches[0], device)))
        total_steps = max(1, math.ceil(tokens / per_batch))

    if preserve_sparsity and masks is None:
        masks = capture_sparsity_masks(model)
    masks = masks or {}
    by_name = dict(model.named_parameters())
    for name in masks:
        if name not in by_name:
            raise RecoveryError(f"mask names a parameter the model does not have: {name!r}")
    before = _mask_stats(model, masks)

    peak_lr = (max_lr or lr) if schedule == "one_cycle" else lr
    opt = optimizer or torch.optim.AdamW(
        _param_groups(model, peak_lr, head_lr_multiplier), weight_decay=weight_decay)
    if schedule == "one_cycle":
        factor, warmup_steps = _one_cycle(total_steps, pct_start)
    else:
        factor, warmup_steps = _warmup_cosine(total_steps, warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)

    handles = []
    if masks and enforce_masks:
        for name, m in masks.items():
            p = by_name[name]
            handles.append(p.register_hook(lambda g, m=m: g * m))

    params = [p for p in model.parameters() if p.requires_grad]
    amp_requested = amp_dtype
    amp_dtype, needs_scaler, amp_note = select_precision(device.type, amp_dtype)
    if amp_requested is not None:
        # One line, naming what actually ran. A run that silently used a
        # different precision from the archived one is not comparable with it,
        # and nothing in the output would have said so.
        logger.warning("recovery precision: %s", amp_note)
    autocast = autocast_for(device.type, amp_dtype)
    scaler = (torch.amp.GradScaler(device.type) if needs_scaler else None)
    loss_curve: list = []
    grad_norms: list = []
    lr_curve: list = []
    epoch_losses: list = []
    tokens_seen = 0
    step = 0
    model.to(device).train()
    try:
        done = False
        while not done:
            running, n = 0.0, 0
            for batch in batches:
                if step >= total_steps:
                    done = True
                    break
                batch = _to_dev(batch, device)
                tokens_seen += _tokens_in(batch)
                with autocast:
                    out = (model(**batch) if isinstance(batch, dict)
                           else model(batch))
                    loss = out.loss if hasattr(out, "loss") else out
                if loss is None:
                    raise RecoveryError(
                        "the model returned no loss for a recovery batch: the "
                        "batches must carry whatever the model needs to compute "
                        "one (labels, for most architectures)")
                if not torch.isfinite(loss):
                    raise RecoveryError(
                        f"non-finite loss at step {step}: {loss.item()}. A diverged "
                        "run still yields a number and writing it is worse than "
                        "stopping.")
                # clip_grad_norm_ returns the norm BEFORE clipping, so the
                # log is the real gradient and not the clipped one. With no
                # clipping asked for, an infinite bound measures without
                # scaling anything.
                lr_curve.append(float(opt.param_groups[0]["lr"]))
                grad_norms.append(backward_and_clip(
                    loss, params, max_grad_norm, opt, scaler))
                opt.zero_grad(set_to_none=True)
                # Not after the final optimizer step. Both schedules are
                # unclamped past the planned end, exactly as torch's is, where
                # the cosine turns back upward; torch forbids the step outright
                # and this loop simply does not take it.
                if step < total_steps - 1:
                    sched.step()
                loss_curve.append(float(loss.detach()))
                running += loss_curve[-1]
                n += 1
                step += 1
            if n:
                epoch_losses.append(running / n)
            if epochs is not None and len(epoch_losses) >= epochs:
                done = True
            if step >= total_steps:
                done = True
    finally:
        for h in handles:
            h.remove()
        model.eval()

    # Measure BEFORE the re-apply. The re-apply would hide exactly the failure
    # this is here to catch: a run whose zeros were filled back in looks clean
    # once they are zeroed again, and the model would then be reported as
    # sparse-and-preserved on the strength of a repair rather than of the run.
    after = _mask_stats(model, masks)
    drift = before["zero_fraction"] - after["zero_fraction"]
    preserved = after["escaped_zeros"] == 0 and abs(drift) == 0.0
    # The re-apply itself is cheap and makes the invariant unconditional: weight
    # decay and momentum are already neutral at zero, but "already neutral" is a
    # property of the optimizer, and the invariant should not depend on which
    # optimizer the caller passed.
    with torch.no_grad():
        for name, m in masks.items():
            by_name[name].data.mul_(m)

    rise, rise_step = max_rise_above_running_min(loss_curve)
    record = {
        "domain": domain,
        "epochs": epochs, "tokens": tokens, "tokens_seen": tokens_seen,
        "steps": step, "total_steps_planned": total_steps,
        "batch_size": batch_size, "lr": lr,
        "head_lr_multiplier": head_lr_multiplier,
        "max_grad_norm": max_grad_norm,
        "amp_dtype": (None if amp_dtype is None else str(amp_dtype)),
        "amp_requested": (None if amp_requested is None else str(amp_requested)),
        "amp_note": amp_note,
        "grad_scaler": bool(needs_scaler),
        "schedule": schedule, "warmup_steps": warmup_steps,
        "peak_lr": peak_lr,
        "warmup_ratio": warmup_ratio if schedule == "warmup_cosine" else None,
        "pct_start": pct_start if schedule == "one_cycle" else None,
        "epoch_losses": epoch_losses,
        "loss_curve": loss_curve, "grad_norms": grad_norms, "lr_curve": lr_curve,
        "loss_first": loss_curve[0] if loss_curve else None,
        "loss_last": loss_curve[-1] if loss_curve else None,
        "loss_min": min(loss_curve) if loss_curve else None,
        "max_rise_above_running_min": rise,
        "max_rise_step": rise_step,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "preserve_sparsity": preserve_sparsity,
        "masks_enforced": bool(masks) and enforce_masks,
        "n_masked_params": len(masks),
        "zero_fraction_before": before["zero_fraction"],
        "zero_fraction_after": after["zero_fraction"],
        "escaped_zeros": after["escaped_zeros"],
        "sparsity_drift": drift,
        "mask_preserved": preserved if masks else None,
    }

    if masks and preserve_sparsity and not preserved:
        raise RecoveryError(
            f"recovery did not preserve the mask: sparsity_drift={drift:.6g}, "
            f"{after['escaped_zeros']} weight(s) that were zero are not any "
            "more. A model that reports itself sparse and is not is worse than "
            "one that failed. Measured drift across 24 of 24 archived recovery "
            "runs was exactly 0.0, so this is a defect and not a tolerance.")
    return record
