"""Tests for the recovery stage. Every one of these runs the code it covers.

Cardinal rule 11: asserting that a piece of source text is present is not a
test of it. So the clipping test reads the recorded norms and the parameter
delta, the schedule test reads the recorded learning-rate curve, and the
mask tests run the real loop over a real mask and then count the zeros.
"""
import pytest
import torch

from sal.pipeline import CompressionPipeline, PipelineError
from sal.recovery import (GENERIC_DOMAIN_WARNING, RecoveryControl, RecoveryError,
                          capture_sparsity_masks, max_rise_above_running_min,
                          recover_model)


def _data(n=6, bs=4, sl=16):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(0, 100, (bs, sl)),
             "labels": torch.randint(0, 100, (bs, sl))} for _ in range(n)]


def _pipe(model, **kw):
    return CompressionPipeline(model, _data(), metric="loss", batch_size=4, **kw)


def _zero_a_block(model, fraction=0.5):
    """Make one 2-D weight genuinely sparse, and return its mask by name."""
    for name, p in model.named_parameters():
        if p.dim() == 2 and p.shape[0] >= 8:
            with torch.no_grad():
                keep = torch.ones_like(p, dtype=torch.bool)
                keep[: int(p.shape[0] * fraction)] = False
                p.data.mul_(keep)
            return {name: keep}
    raise AssertionError("fixture has no 2-D weight to sparsify")


# --------------------------------------------------------------- stage ordering
def test_recover_before_compress_raises(tiny_model):
    pipe = _pipe(tiny_model)
    with pytest.raises(PipelineError, match="compress"):
        pipe.recover(_data(), domain="same as eval")


def test_epochs_defaults_to_two_and_lands_in_the_record(tiny_model):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(_data(), domain="same as eval")
    rec = pipe.validate().recovery
    assert rec["epochs"] == 2
    assert rec["steps"] == rec["total_steps_planned"] == 2 * len(_data())
    assert "2 epoch(s)" in pipe.validate().stages[-1].detail


def test_loss_decreases_over_two_epochs(tiny_model):
    rec = recover_model(tiny_model, _data(), epochs=2, lr=1e-3, batch_size=4, domain="same as eval")
    assert len(rec["epoch_losses"]) == 2
    assert rec["epoch_losses"][1] < rec["epoch_losses"][0]


def test_exactly_one_budget(tiny_model):
    with pytest.raises(ValueError, match="exactly one budget"):
        recover_model(tiny_model, _data(), epochs=2, tokens=1000, domain="same as eval")
    with pytest.raises(ValueError, match="exactly one budget"):
        recover_model(tiny_model, _data(), epochs=None, tokens=None, domain="same as eval")


def test_token_budget_builds_its_own_schedule(tiny_model):
    # 4x16 = 64 tokens a batch, so 200 tokens is four steps.
    rec = recover_model(tiny_model, _data(), epochs=None, tokens=200, batch_size=4, domain="same as eval")
    assert rec["total_steps_planned"] == 4
    assert rec["steps"] == 4
    assert rec["tokens_seen"] >= 200
    assert rec["epochs"] is None and rec["tokens"] == 200


# ------------------------------------------------------------------- sparsity
def test_sparsity_survives_a_user_supplied_mask(tiny_model):
    """The rule-11 test for the mask machinery v0.7.0 ships without a pruner."""
    masks = _zero_a_block(tiny_model)
    name, keep = next(iter(masks.items()))
    param = dict(tiny_model.named_parameters())[name]
    zeros_before = (param.data == 0).clone()

    rec = recover_model(tiny_model, _data(), epochs=2, lr=1e-2, batch_size=4,
                        masks=masks, preserve_sparsity=True, domain="same as eval")

    assert rec["n_masked_params"] == 1
    assert rec["sparsity_drift"] == 0.0
    assert rec["mask_preserved"] is True
    assert rec["escaped_zeros"] == 0
    assert torch.equal((param.data == 0), zeros_before)
    assert bool((param.data[~keep] == 0).all())
    # and the kept half did train, or the test would pass on a frozen model
    assert bool((param.data[keep] != 0).any())


def test_sparsity_loss_raises(tiny_model):
    """With the gradient hook off, the same run must raise, not report sparse."""
    masks = _zero_a_block(tiny_model)
    with pytest.raises(RecoveryError, match="did not preserve the mask"):
        recover_model(tiny_model, _data(), epochs=2, lr=1e-2, batch_size=4,
                      masks=masks, preserve_sparsity=True, enforce_masks=False, domain="same as eval")


def test_mask_names_an_absent_parameter_raises(tiny_model):
    with pytest.raises(RecoveryError, match="does not have"):
        recover_model(tiny_model, _data(), epochs=1,
                      masks={"no.such.weight": torch.ones(2, 2, dtype=torch.bool)}, domain="same as eval")


def test_capture_finds_no_mask_on_a_dense_model(tiny_model):
    assert capture_sparsity_masks(tiny_model) == {}


def test_preserve_sparsity_is_a_noop_on_a_sliced_model(tiny_model):
    """A structurally sliced model has no mask, so the flag must not error."""
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None, slice_heads=True)
    pipe.recover(_data(), domain="same as eval", epochs=1, preserve_sparsity=True)
    rec = pipe.validate().recovery
    assert rec["n_masked_params"] == 0
    assert rec["mask_preserved"] is None
    assert rec["sparsity_drift"] == 0.0


# -------------------------------------------------------------------- clipping
def test_clipping_bounds_the_step(tiny_model):
    """Run it twice on the same weights: the clipped run must move less.

    Under SGD, where the step is proportional to the gradient it was given.
    AdamW normalises by the gradient's own magnitude, so a clip changes its
    step size very little; that is a property of Adam and not an absence of
    clipping, and testing the effect under AdamW would measure the optimizer.
    """
    import copy

    data = _data(n=1)
    clipped, loose = copy.deepcopy(tiny_model), copy.deepcopy(tiny_model)
    start = {k: v.clone() for k, v in tiny_model.state_dict().items()}

    rl = recover_model(loose, data, epochs=1, batch_size=4, max_grad_norm=None,
                       optimizer=torch.optim.SGD(loose.parameters(), lr=0.1), domain="same as eval")
    # Clip at half the norm this fixture actually produces. A bound orders of
    # magnitude below it would scale the update below float32's resolution at
    # these weights, and the test would then be measuring rounding.
    bound = rl["grad_norms"][0] / 2
    rc = recover_model(clipped, data, epochs=1, batch_size=4, max_grad_norm=bound,
                       optimizer=torch.optim.SGD(clipped.parameters(), lr=0.1), domain="same as eval")

    # The logged norm is the norm BEFORE clipping, so both runs see the same one.
    assert rc["grad_norms"][0] == pytest.approx(rl["grad_norms"][0], rel=1e-5)
    assert rc["grad_norms"][0] > bound, "the fixture must exceed the clip bound"
    assert rc["max_grad_norm"] == bound and rl["max_grad_norm"] is None

    def moved(model):
        now = model.state_dict()
        return sum(float((now[k] - v).abs().sum()) for k, v in start.items())

    # The clip scales the whole gradient by bound/(norm + eps), so the ratio of
    # the two movements is that factor rather than merely "smaller".
    assert moved(clipped) < moved(loose)
    ratio = bound / (rc["grad_norms"][0] + 1e-6)
    assert ratio == pytest.approx(0.5, rel=1e-3)
    assert moved(clipped) == pytest.approx(moved(loose) * ratio, rel=1e-3)


def test_grad_norms_and_loss_curve_have_one_entry_per_step(tiny_model):
    rec = recover_model(tiny_model, _data(n=3), epochs=2, batch_size=4, domain="same as eval")
    assert rec["steps"] == 6
    assert len(rec["grad_norms"]) == 6
    assert len(rec["loss_curve"]) == 6
    assert len(rec["lr_curve"]) == 6
    assert rec["grad_norm_max"] == max(rec["grad_norms"])


def test_non_finite_loss_raises(tiny_model):
    class Diverged(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, **kw):
            out = self.inner(**kw)
            out.loss = out.loss * float("inf")
            return out

    with pytest.raises(RecoveryError, match="non-finite loss"):
        recover_model(Diverged(tiny_model), _data(n=1), epochs=1, batch_size=4, domain="same as eval")


# -------------------------------------------------------------------- schedule
def test_warmup_does_not_peak_outside_its_window(tiny_model):
    rec = recover_model(tiny_model, _data(n=10), epochs=1, batch_size=4,
                        schedule="warmup_cosine", warmup_ratio=0.5, domain="same as eval")
    lrs = rec["lr_curve"]
    peak = lrs.index(max(lrs))
    assert lrs[0] < max(lrs), "step 0 must not already be at the peak"
    assert peak <= rec["warmup_steps"], (peak, rec["warmup_steps"])


def test_one_cycle_climbs_to_a_peak_above_its_start(tiny_model):
    """The published decoder shape, and the reason the gate has two modes.

    It starts at a twenty-fifth of the peak and climbs for the first stretch of
    the run, which is the shape a recovery diverged under: the learning rate
    was still rising while the model was at its most damaged.
    """
    from sal.recovery import ONE_CYCLE_DIV_FACTOR

    rec = recover_model(tiny_model, _data(n=10), epochs=1, batch_size=4,
                        schedule="one_cycle", max_lr=1e-4, pct_start=0.30, domain="same as eval")
    lrs = rec["lr_curve"]
    peak = lrs.index(max(lrs))
    assert rec["peak_lr"] == 1e-4
    assert lrs[0] == pytest.approx(1e-4 / ONE_CYCLE_DIV_FACTOR)
    assert max(lrs) == pytest.approx(1e-4)
    # torch's up phase ends at pct_start * total_steps - 1, so with 10 steps
    # at 0.30 the peak is step 2 and not step 3.
    assert peak == rec["warmup_steps"] == 2
    assert lrs[0] < lrs[1] < lrs[2], "it must still be climbing"
    # It is still decaying at the last recorded step: the floor is only reached
    # at the planned final step, which is one past the last one that runs.
    assert lrs[-1] < max(lrs) / 10, "and then come down a long way"
    assert rec["pct_start"] == 0.30 and rec["warmup_ratio"] is None


def test_warmup_cosine_never_exceeds_the_base_rate(tiny_model):
    """The contrast with one_cycle: no excursion above the rate asked for."""
    rec = recover_model(tiny_model, _data(n=10), epochs=1, batch_size=4,
                        lr=5e-5, schedule="warmup_cosine", warmup_ratio=0.5, domain="same as eval")
    assert max(rec["lr_curve"]) == pytest.approx(5e-5)
    assert rec["peak_lr"] == 5e-5


def test_unknown_schedule_raises(tiny_model):
    with pytest.raises(ValueError, match="schedule must be"):
        recover_model(tiny_model, _data(n=1), epochs=1, schedule="cosine", domain="same as eval")


# --------------------------------------------------------------------- the rise
def test_max_rise_above_running_min():
    assert max_rise_above_running_min([5, 4, 3, 2, 1]) == (0.0, None)
    assert max_rise_above_running_min([5, 3, 4, 2, 9, 2]) == (7.0, 4)
    assert max_rise_above_running_min([]) == (0.0, None)


def test_the_rise_is_recorded_with_the_curve(tiny_model):
    rec = recover_model(tiny_model, _data(n=3), epochs=1, batch_size=4, domain="same as eval")
    rise, step = max_rise_above_running_min(rec["loss_curve"])
    assert rec["max_rise_above_running_min"] == rise
    assert rec["max_rise_step"] == step


# ----------------------------------------------------------------------- domain
def test_domain_is_required(tiny_model):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    with pytest.raises(TypeError):
        pipe.recover(_data())
    with pytest.raises(PipelineError, match="domain="):
        pipe.recover(_data(), domain="  ")


def test_domain_reaches_the_record(tiny_model):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(_data(), domain="wikitext-2 train", epochs=1)
    assert pipe.validate().recovery["domain"] == "wikitext-2 train"


def test_generic_domain_warns_with_the_measured_reason(tiny_model, caplog):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    with caplog.at_level("WARNING"):
        pipe.recover(_data(), domain="generic", epochs=1)
    assert any("0.6398" in r.message for r in caplog.records)
    assert "3 of 3 seeds" in GENERIC_DOMAIN_WARNING


def test_domain_guard_fires_when_the_control_moved_as_far(tiny_model, caplog):
    """metric='loss': the control is worse than the original by more than the
    arm's one-shot damage, which is the OPT-6.7B shape."""
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    original = pipe.original_accuracy
    one_shot = pipe._one_shot_accuracy
    damage = one_shot - original
    worse_control = original + damage * 2 + 1.0
    with caplog.at_level("WARNING"):
        pipe.recover(_data(), domain="generic", epochs=1,
                     control=RecoveryControl(worse_control, epochs=1, metric="loss"))
    rec = pipe.validate().recovery
    assert rec["domain_shift_dominates"] is True
    assert any("domain shift" in r.message for r in caplog.records)


def test_domain_guard_quiet_when_the_control_held(tiny_model):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    good_control = pipe.original_accuracy  # the control did not move at all
    pipe.recover(_data(), domain="same as eval", epochs=1,
                 control=RecoveryControl(good_control, epochs=1, metric="loss"))
    assert pipe.validate().recovery["domain_shift_dominates"] is False


# ------------------------------------------------------------------- the floor
def _staged_losses(pipe, values):
    """Control what each stage measures, so the floor's own logic is the test.

    The floor is a state machine over stage scores: held while the model is
    still repairable, applied at validate(), and blamed on the ratio once
    recovery has run. Driving it with the fixture's real loss would test
    whether pruning 25% of a random tiny model happens to raise its loss past
    a threshold, which is not the behaviour under test. The evaluator itself is
    covered by the stages in the tests above.
    """
    it = iter(values)
    pipe._accuracy = lambda model=None: next(it)
    return pipe


def test_floor_is_held_at_compress_and_names_recovery_at_validate(tiny_model):
    # metric="loss", so the floor is an upper bound: 2.0 is past 1.5, 1.0 is not.
    pipe = _staged_losses(_pipe(tiny_model, accuracy_floor=1.5), [1.0, 2.0])
    pipe.compress(pruning=0.25, quantization=None)      # held, not raised
    assert pipe._deferred_floor is not None
    with pytest.raises(PipelineError, match=r"recover\(\) before validate"):
        pipe.validate()


def test_a_held_floor_is_cleared_by_a_recovery_that_passes(tiny_model):
    pipe = _staged_losses(_pipe(tiny_model, accuracy_floor=1.5),
                          [1.0, 2.0, 1.1])
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(_data(), domain="same as eval", epochs=1)
    assert pipe._deferred_floor is None
    assert pipe.validate().stages[-1].accuracy == 1.1


def test_floor_after_recovery_blames_the_ratio(tiny_model):
    pipe = _staged_losses(_pipe(tiny_model, accuracy_floor=1.5),
                          [1.0, 2.0, 1.9])
    pipe.compress(pruning=0.25, quantization=None)
    with pytest.raises(PipelineError, match="prune_ratio"):
        pipe.recover(_data(), domain="same as eval", epochs=1)


def test_strict_recovery_refuses_to_score_an_unrecovered_model(tiny_model):
    pipe = _pipe(tiny_model, strict_recovery=True)
    pipe.compress(pruning=0.25, quantization=None)
    with pytest.raises(PipelineError, match="strict_recovery"):
        pipe.validate()


def test_strict_recovery_passes_once_recovered(tiny_model):
    pipe = _pipe(tiny_model, strict_recovery=True)
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(_data(), domain="same as eval", epochs=1)
    assert pipe.validate().stages[-1].name == "recovered"


# ------------------------------------------------------------- the report field
def test_recovery_record_serializes(tiny_model):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(_data(), domain="same as eval", epochs=1)
    d = pipe.validate().to_dict()
    assert d["recovery"]["domain"] == "same as eval"
    assert len(d["recovery"]["grad_norms"]) == d["recovery"]["steps"]
    import json
    json.loads(json.dumps(d))     # the record must survive a round trip


def test_control_budget_matching():
    assert RecoveryControl(0.8, epochs=2).matches(2, None) is True
    assert RecoveryControl(0.8, epochs=5).matches(2, None) is False
    assert RecoveryControl(0.8, tokens=32_000_000).matches(None, 32_000_000) is True
    assert RecoveryControl(0.8, tokens=32_000_000).matches(None, 16_000_000) is False
    assert RecoveryControl(0.8, epochs=2).matches(None, 32_000_000) is False
    assert RecoveryControl(0.8).matches(2, None) is False


# -------------------------------------------------------------------- autocast
def test_autocast_for_returns_a_real_context_for_an_accelerator():
    """The branch a CPU-only suite cannot otherwise reach.

    Asserting that the source mentions autocast would be a source test, which
    rule 11 allows only as a supplement. The choice is a function of the device
    type, so the CUDA branch is executable here even with no CUDA device: what
    is checked is which context object comes back, not that it casts.
    """
    import contextlib

    from sal.recovery import autocast_for

    cpu = autocast_for("cpu", torch.bfloat16)
    assert isinstance(cpu, contextlib.nullcontext), (
        "CPU autocast is a different dtype policy from the archived runs', so "
        "the flag must not be honoured there")
    off = autocast_for("cuda", None)
    assert isinstance(off, contextlib.nullcontext)

    on = autocast_for("cuda", torch.bfloat16)
    assert isinstance(on, torch.autocast)
    assert on.device == "cuda" and on.fast_dtype == torch.bfloat16

    named = autocast_for("cuda", "float16")
    assert isinstance(named, torch.autocast)
    assert named.fast_dtype == torch.float16


def test_the_recovery_record_states_the_autocast_it_ran_under(tiny_model):
    """On a CPU a pin falls back, and the record says what actually ran."""
    rec = recover_model(tiny_model, _data(n=2), epochs=1, batch_size=4,
                        amp_dtype=torch.bfloat16, domain="same as eval")
    assert rec["amp_requested"] == "torch.bfloat16"
    assert rec["amp_dtype"] is None
    assert "CPU" in rec["amp_note"] and "pinned" in rec["amp_note"]
    off = recover_model(tiny_model, _data(n=2), epochs=1, batch_size=4, domain="same as eval")
    assert off["amp_dtype"] is None and off["amp_requested"] is None


def test_the_pipeline_records_the_precision_it_scored_at(tiny_model):
    pipe = _pipe(tiny_model)
    assert pipe.precision["eval_autocast"] is None
    pipe2 = CompressionPipeline(tiny_model, _data(), metric="loss",
                                batch_size=4, eval_autocast="float16")
    assert pipe2.precision["eval_autocast"] == "float16"
    # On CPU it is a no-op, so the score is the same either way.
    assert pipe2._accuracy() == pytest.approx(pipe._accuracy())


def test_domain_is_required_by_the_loop_too(tiny_model):
    """Not only by the pipeline wrapper: the gate drives this function."""
    with pytest.raises(TypeError):
        recover_model(tiny_model, _data(), epochs=1)   # no domain at all
    with pytest.raises(ValueError, match="needs domain="):
        recover_model(tiny_model, _data(), domain="   ", epochs=1)
    rec = recover_model(tiny_model, _data(n=1), domain="wikitext-2 train",
                        epochs=1, batch_size=4)
    assert rec["domain"] == "wikitext-2 train"


def test_generic_warns_from_the_loop(tiny_model, caplog):
    with caplog.at_level("WARNING"):
        recover_model(tiny_model, _data(n=1), domain="generic", epochs=1,
                      batch_size=4)
    assert any("0.6398" in r.message for r in caplog.records)


# ---------------------------------------------------------------- the budget
def test_resolve_budget_tells_a_default_from_a_choice():
    """The trap the gate's micro-test found: tokens= against an epochs default.

    With `epochs=2` as a plain default, a decoder caller passing `tokens=` was
    refused for "passing both", and had to write `epochs=None` to say what the
    token budget already said.
    """
    from sal.recovery import DEFAULT_EPOCHS, UNSET, resolve_budget

    assert resolve_budget(UNSET, None) == (DEFAULT_EPOCHS, None)
    assert resolve_budget(UNSET, 128) == (None, 128)
    assert resolve_budget(3, None) == (3, None)
    assert resolve_budget(None, 128) == (None, 128)
    with pytest.raises(ValueError, match="Both were given"):
        resolve_budget(2, 128)
    with pytest.raises(ValueError, match="got neither"):
        resolve_budget(None, None)
    assert DEFAULT_EPOCHS == 2


def test_token_mode_needs_no_epochs_argument(tiny_model):
    rec = recover_model(tiny_model, _data(n=4), domain="same as eval",
                        tokens=200, batch_size=4)
    assert rec["epochs"] is None and rec["tokens"] == 200
    assert rec["steps"] == 4


def test_the_pipeline_defaults_to_two_epochs_and_accepts_tokens(tiny_model):
    pipe = _pipe(tiny_model)
    pipe.compress(pruning=0.25, quantization=None)
    pipe.recover(_data(), domain="same as eval")
    assert pipe.validate().recovery["epochs"] == 2

    pipe2 = _pipe(tiny_model)
    pipe2.compress(pruning=0.25, quantization=None)
    pipe2.recover(_data(), domain="same as eval", tokens=200)
    rec = pipe2.validate().recovery
    assert rec["epochs"] is None and rec["tokens"] == 200
    assert "200 tokens" in pipe2.validate().stages[-1].detail


# ------------------------------------------------- device-aware precision
def test_select_precision_covers_every_device_and_request():
    """Each branch, with the device's capability injected rather than probed.

    The float16-with-a-scaler branch cannot run on this machine, so what is
    tested here is the decision: which dtype, whether a scaler is needed, and
    what the note says. The call sequence that branch would take is tested
    separately in test_backward_and_clip_unscales_before_it_measures.
    """
    from sal.recovery import ARCHIVED_AMP, select_precision

    assert select_precision("cpu", None) == (None, False, "fp32, no autocast")
    assert select_precision("cuda", None)[0] is None

    dtype, scaler, note = select_precision("cuda", "auto", supports_bf16=True)
    assert (dtype, scaler) == (torch.bfloat16, False)
    assert "as the archive" in note

    dtype, scaler, note = select_precision("cuda", "auto", supports_bf16=False)
    assert (dtype, scaler) == (torch.float16, True)
    assert "GradScaler" in note and ARCHIVED_AMP in note

    dtype, scaler, note = select_precision("cpu", "auto", supports_bf16=True)
    assert (dtype, scaler) == (None, False)
    assert "fp32 on a CPU" in note and ARCHIVED_AMP in note

    # A pin the device can honour is honoured.
    assert select_precision("cuda", torch.bfloat16, supports_bf16=True)[:2] == \
        (torch.bfloat16, False)
    assert select_precision("cuda", "float16", supports_bf16=True)[:2] == \
        (torch.float16, True)


def test_a_pinned_bfloat16_is_refused_where_it_is_unsupported():
    """Pinning is for reproducing a cell, so a substitution would be a lie."""
    from sal.recovery import select_precision

    with pytest.raises(RecoveryError, match="pinned"):
        select_precision("cuda", torch.bfloat16, supports_bf16=False)

    # On a CPU it falls back instead of raising: nobody reproduces a published
    # accelerator figure on a CPU, and the gate's own micro-test runs here.
    dtype, scaler, note = select_precision("cpu", torch.bfloat16,
                                           supports_bf16=True)
    assert dtype is None and scaler is False
    assert "pinned" in note and "CPU" in note


def test_backward_and_clip_unscales_before_it_measures():
    """The order a scaler forces, checked by recording the calls.

    With a scaler the gradients have to be unscaled before they are clipped or
    measured, or the logged norm is the loss scale rather than the gradient.
    This runs the real function against a recording stub, so the ordering is
    executed rather than read.
    """
    from sal.recovery import backward_and_clip

    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = []

    class Stub:
        def scale(self, loss):
            calls.append("scale")
            return loss

        def unscale_(self, optimizer):
            calls.append("unscale_")

        def step(self, optimizer):
            calls.append("step")
            optimizer.step()

        def update(self):
            calls.append("update")

    loss = model(torch.randn(2, 4)).sum()
    norm = backward_and_clip(loss, list(model.parameters()), 1.0, opt, Stub())
    assert calls == ["scale", "unscale_", "step", "update"]
    assert norm > 0

    # And without a scaler the plain path still steps the optimizer once.
    opt2 = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss2 = model(torch.randn(2, 4)).sum()
    backward_and_clip(loss2, list(model.parameters()), 1.0, opt2, None)
    assert not torch.equal(before, model.weight.detach())


def test_auto_precision_warns_with_the_precision_it_used(tiny_model, caplog):
    with caplog.at_level("WARNING"):
        rec = recover_model(tiny_model, _data(n=1), domain="same as eval",
                            epochs=1, batch_size=4, amp_dtype="auto")
    assert rec["amp_requested"] == "auto"
    assert rec["grad_scaler"] is False
    assert rec["amp_dtype"] is None, "CPU: fp32, whatever was asked for"
    line = [r.getMessage() for r in caplog.records
            if "recovery precision" in str(r.msg)]
    assert line and "fp32 on a CPU" in line[0]
    assert "bfloat16" in line[0], "the warning must name the archived precision"


def test_no_warning_when_no_autocast_was_asked_for(tiny_model, caplog):
    with caplog.at_level("WARNING"):
        rec = recover_model(tiny_model, _data(n=1), domain="same as eval",
                            epochs=1, batch_size=4)
    assert rec["amp_note"] == "fp32, no autocast"
    assert not [r for r in caplog.records if "recovery precision" in str(r.msg)]
