"""`recover()` must stop on an absurd gradient, not clip it and carry on.

The B3 pilot's arms B and C each ran 1221 steps with pre-clip gradient norms
between 1e14 and 1e15 -- every step past any sane bound -- inside the
SHIPPED `recover_model`. Both finished, both reported a loss curve that
moved, and neither model had learned: arm B's accuracy after 10M tokens of
fine-tuning was 52.97% against the untrained 53.23%. `recover_model` already
recorded `grad_norm_max` and nothing read it, and it already raised on a
non-finite LOSS, which a finite loss with an exploded gradient sails past.

Every test here RUNS the loop (rule 11). The explosion is produced by a real
model whose weights make a real gradient enormous, not by patching the clip
or injecting a number, because the thing under test is whether the loop
notices what the loop computes.
"""
import re

import pytest
import torch
import torch.nn as nn

from sal.recovery import (GRADIENT_NORM_BOUND, RecoveryError,
                          check_gradient_norm, describe_attention,
                          recover_model)


def _data(n=4, bs=2, sl=8, vocab=16):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(0, vocab, (bs, sl)),
             "labels": torch.randint(0, vocab, (bs, sl))} for _ in range(n)]


class _Cfg:
    model_type = "toytransformer"
    _attn_implementation = "sdpa"


class _Exploder(nn.Module):
    """A real model whose real gradient is enormous, by a chosen factor.

    `scale` multiplies the logits, so the cross-entropy gradient scales with
    it and the norm is genuinely large rather than asserted to be. At
    `scale=1` the same model trains normally, which is what makes the
    comparison in these tests mean anything.
    """

    def __init__(self, vocab=16, hidden=8, scale=1.0):
        super().__init__()
        self.config = _Cfg()
        self.scale = scale
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.head.weight.mul_(1.0)

    def forward(self, input_ids=None, labels=None, **kw):
        logits = self.head(self.embed(input_ids)) * self.scale
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        class Out:
            pass

        o = Out()
        o.logits, o.loss = logits, loss
        return o


# ------------------------------------------------------------- the bound
def test_the_bound_is_between_the_measured_healthy_and_failing_norms():
    """1e4 is a measurement, not a taste: 20x the worst healthy, 1e10 under
    the failure."""
    assert GRADIENT_NORM_BOUND == pytest.approx(1.0e4)
    healthy_worst = 610.4        # SDPA, fp32, probe job 23937845
    failing = 6.93e14            # SDPA, bf16 autocast, same batches
    assert healthy_worst < GRADIENT_NORM_BOUND < failing
    assert GRADIENT_NORM_BOUND / healthy_worst > 10
    assert failing / GRADIENT_NORM_BOUND > 1e9


# ---------------------------------------------------------- the guard alone
def test_a_norm_inside_the_bound_passes():
    check_gradient_norm(1.0, 0, _Exploder(), torch.bfloat16)
    check_gradient_norm(GRADIENT_NORM_BOUND, 0, _Exploder(), torch.bfloat16)


def test_a_norm_past_the_bound_raises_and_names_the_attention():
    with pytest.raises(RecoveryError) as exc:
        check_gradient_norm(6.93e14, 7, _Exploder(), torch.bfloat16)
    msg = str(exc.value)
    assert "step 7" in msg
    assert "6.93e+14" in msg
    assert "'sdpa'" in msg, msg
    assert "bfloat16" in msg
    assert "toytransformer" in msg
    assert "eager" in msg
    assert "grad_norm_bound=" in msg


def test_a_non_finite_norm_raises_even_though_the_loss_was_finite():
    """The gap the old guard left: it checked the loss, not the gradient."""
    for bad in (float("inf"), float("nan"), float("-inf")):
        with pytest.raises(RecoveryError, match="not finite"):
            check_gradient_norm(bad, 0, _Exploder(), None)


def test_the_message_quotes_the_measured_comparison():
    """So a reader does not have to find the probe to know what to change."""
    with pytest.raises(RecoveryError) as exc:
        check_gradient_norm(1e15, 0, _Exploder(), torch.bfloat16)
    msg = str(exc.value)
    assert "498.7" in msg and "610.4" in msg and "6.93e14" in msg
    assert "MI250X" in msg


def test_describe_attention_says_so_when_the_config_is_silent():
    class Bare(nn.Module):
        pass

    info = describe_attention(Bare())
    assert "unknown" in info["attn_implementation"]
    assert info["model_type"] is None


# ------------------------------------------------- the guard inside the loop
def test_recover_model_stops_on_the_first_exploded_step():
    """Executed end to end: a real gradient, a real loop, a real refusal."""
    model = _Exploder(scale=1e7)
    with pytest.raises(RecoveryError) as exc:
        recover_model(model, _data(), epochs=1, lr=1e-4, batch_size=2,
                      domain="same as eval", preserve_sparsity=False)
    msg = str(exc.value)
    assert "BEFORE clipping" in msg
    step = int(re.search(r"step (\d+)", msg).group(1))
    assert step == 0, (
        "the guard must fire on the FIRST offending step: a guard that "
        "reports at the end has already spent the budget. Fired at %d" % step)


def test_the_same_model_trains_normally_without_the_explosion():
    """The comparison that makes the test above mean something."""
    model = _Exploder(scale=1.0)
    rec = recover_model(model, _data(), epochs=1, lr=1e-3, batch_size=2,
                        domain="same as eval", preserve_sparsity=False)
    assert rec["grad_norm_max"] < GRADIENT_NORM_BOUND
    # `_data()` hands the loop four ready-made batches and each is one step;
    # `batch_size` does not re-batch them.
    assert rec["steps"] == 4
    assert rec["grad_norm_bound"] == GRADIENT_NORM_BOUND


def test_a_caller_who_knows_better_can_raise_the_bound():
    """`domain="generic"` sets the precedent: proceed knowingly, not blindly."""
    model = _Exploder(scale=1e7)
    rec = recover_model(model, _data(), epochs=1, lr=1e-12, batch_size=2,
                        domain="same as eval", preserve_sparsity=False,
                        grad_norm_bound=float("inf"))
    assert rec["grad_norm_max"] > GRADIENT_NORM_BOUND
    assert rec["grad_norm_bound"] == float("inf")


def test_the_record_carries_the_attention_implementation():
    """So a result says what it ran under, not only what it measured."""
    rec = recover_model(_Exploder(), _data(), epochs=1, lr=1e-3,
                        batch_size=2, domain="same as eval",
                        preserve_sparsity=False)
    assert rec["attention"]["attn_implementation"] == "sdpa"
    assert rec["attention"]["model_type"] == "toytransformer"


def test_the_guard_fires_with_no_clipping_at_all():
    """The published decoder recipe is UNCLIPPED, so this is the case that
    matters most.

    `exp36`'s recovery and v0.7.0's G3-paper arms run `max_grad_norm=None`,
    and the archived 6.7B `ffn_random_50` anomaly -- a loss rising 4.08 to
    5.43 nats above its own running minimum, at steps that are not its
    seed's data step -- ran that way too, with gradient norms that the record
    itself says "were not logged at the time". A guard that only worked when
    clipping is on would miss exactly the configuration in which an
    explosion moves the weights instead of freezing them.

    `backward_and_clip` passes an infinite bound to `clip_grad_norm_` when no
    clipping is asked for, which still returns the true norm, so the guard
    has the number either way.
    """
    model = _Exploder(scale=1e7)
    with pytest.raises(RecoveryError) as exc:
        recover_model(model, _data(), epochs=1, lr=1e-4, batch_size=2,
                      domain="same as eval", preserve_sparsity=False,
                      max_grad_norm=None)
    msg = str(exc.value)
    assert "BEFORE clipping" in msg
    assert int(re.search(r"step (\d+)", msg).group(1)) == 0


def test_the_bound_clears_every_archived_recovery():
    """A bound that retroactively condemned healthy published work would be
    the wrong bound.

    The v0.7.0 gate's six OPT-1.3B arms ran under bfloat16 autocast -- the
    same precision as the failing Qwen2.5 combination -- and recorded
    pre-clip maxima of 46.81, 83.54, 128.3, 223.0, 612.3 and 992.6, with
    every loss curve falling. The worst of them is more than ten times below
    this bound.
    """
    archived = [46.81, 83.54, 128.3, 223.0, 612.3, 992.6]
    assert max(archived) < GRADIENT_NORM_BOUND
    assert GRADIENT_NORM_BOUND / max(archived) > 10
    for norm in archived:
        check_gradient_norm(norm, 0, _Exploder(), torch.bfloat16)
