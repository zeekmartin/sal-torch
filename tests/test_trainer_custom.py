"""SALTrainer with a custom training step (v0.5.1).

The point of the custom callback is that SAL stops being a supervised-only
tool: the masking mechanism is identical, only the loss moves out of the model
and into the caller. These tests pin both halves of that — the callback really
does drive training, and the no-callback path is byte-for-byte the old
behaviour.
"""
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from sal.config import SALConfig
from sal.trainer import SALTrainer


class _DictDS(Dataset):
    """Batches shaped like the supervised path expects (input_ids + labels)."""
    def __init__(self, n=8):
        torch.manual_seed(0)
        self.ids = torch.randint(0, 100, (n, 16))
    def __len__(self):
        return self.ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


@pytest.fixture
def loader():
    return DataLoader(_DictDS(8), batch_size=2)


@pytest.fixture
def cfg():
    # prune_start_ratio=0 so masking is live from the very first step, which
    # keeps these tests independent of the schedule ramp.
    return SALConfig(num_layers=4, num_heads_per_layer=8,
                     attention_pattern="transformer.h.{}.attn",
                     prune_fraction=0.33, prune_start_ratio=0.0,
                     prune_end_ratio=0.5)


def _make(model, cfg, loader, train_step=None):
    return SALTrainer(model, cfg, SGD(model.parameters(), lr=1e-3), loader,
                      seed=7, train_step=train_step)


# --------------------------------------------------------------------- basics
def test_custom_train_step_callback(tiny_model, cfg, loader):
    """The callback is invoked once per batch and its losses are collected."""
    calls = []

    def step(model, batch, optimizer, mask_module):
        calls.append(batch["input_ids"].shape)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    hist = _make(tiny_model, cfg, loader, step).train(num_epochs=2)

    assert len(calls) == len(loader) * 2          # every batch, every epoch
    assert len(hist["losses"]) == 2               # one mean loss per epoch
    assert all(isinstance(x, float) for x in hist["losses"])
    assert hist["total_steps"] == len(loader) * 2


def test_custom_step_actually_trains(tiny_model, cfg, loader):
    """Weights move — the callback's optimizer.step() is the real one."""
    before = tiny_model.head.weight.detach().clone()

    def step(model, batch, optimizer, mask_module):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    _make(tiny_model, cfg, loader, step).train(num_epochs=1)
    assert not torch.allclose(before, tiny_model.head.weight)


def test_custom_step_accepts_tensor_loss(tiny_model, cfg, loader):
    """Returning a 0-d Tensor is as good as returning a float."""
    def step(model, batch, optimizer, mask_module):
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return loss                                # Tensor, not .item()

    hist = _make(tiny_model, cfg, loader, step).train(num_epochs=1)
    assert isinstance(hist["losses"][0], float)


def test_custom_step_bad_return_raises(tiny_model, cfg, loader):
    def step(model, batch, optimizer, mask_module):
        return {"loss": 1.0}                       # not a scalar

    with pytest.raises(TypeError, match="float or a scalar Tensor"):
        _make(tiny_model, cfg, loader, step).train(num_epochs=1)


def test_non_callable_train_step_raises(tiny_model, cfg, loader):
    with pytest.raises(TypeError, match="train_step must be callable"):
        SALTrainer(tiny_model, cfg, SGD(tiny_model.parameters(), lr=1e-3),
                   loader, train_step="not a function")


# ------------------------------------------------------------ mask plumbing
def test_custom_step_receives_mask_module(tiny_model, cfg, loader):
    """The 4th argument is the trainer's live HeadMasker, already installed."""
    from sal.masker import HeadMasker
    seen = []

    def step(model, batch, optimizer, mask_module):
        seen.append(mask_module)
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return loss.item()

    trainer = _make(tiny_model, cfg, loader, step)
    trainer.train(num_epochs=1)

    assert seen, "callback never ran"
    assert all(m is trainer.masker for m in seen)
    assert all(isinstance(m, HeadMasker) for m in seen)


def test_mask_applied_in_custom_step(tiny_model, cfg, loader):
    """Masking is live when the callback is entered, and heads are really zeroed."""
    states, pruned = [], []

    def step(model, batch, optimizer, mask_module):
        states.append(mask_module.masking)
        pruned.append(mask_module.stats["pruned_heads"])
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return loss.item()

    _make(tiny_model, cfg, loader, step).train(num_epochs=2)

    assert all(states), "masker should be active when the callback runs"
    assert max(pruned) > 0, "no heads were ever pruned"
    assert pruned == sorted(pruned), "pruned set must accumulate, never shrink"


def test_mask_removed_after_custom_step(tiny_model, cfg, loader):
    """remove_mask() inside the callback suspends masking without losing heads."""
    after = []

    def step(model, batch, optimizer, mask_module):
        mask_module.apply_mask()
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        mask_module.remove_mask()
        after.append((mask_module.masking, mask_module.stats["pruned_heads"]))
        return loss.item()

    _make(tiny_model, cfg, loader, step).train(num_epochs=2)

    assert all(not active for active, _ in after), "remove_mask() left masking on"
    counts = [n for _, n in after]
    assert counts == sorted(counts), "remove_mask() must not restore pruned heads"
    assert max(counts) > 0


def test_remove_mask_preserves_pruned_set(tiny_model, cfg):
    """The distinction from deactivate(): suspend keeps the mask, reset clears it."""
    from sal.masker import HeadMasker
    m = HeadMasker(tiny_model, cfg, seed=3)
    m.install()
    try:
        m.activate()
        snapshot = {k: v.clone() for k, v in m._masks.items()}
        assert m.stats["pruned_heads"] > 0

        m.remove_mask()
        assert m.masking is False
        assert all(torch.equal(snapshot[k], m._masks[k]) for k in snapshot)

        m.apply_mask()
        assert m.masking is True
        assert all(torch.equal(snapshot[k], m._masks[k]) for k in snapshot)

        m.deactivate()                             # the resetting one
        assert m.stats["pruned_heads"] == 0
    finally:
        m.remove()


def test_unmasked_context_restores_state(tiny_model, cfg):
    """`with masker.unmasked()` is exception-safe and does not turn masking on."""
    from sal.masker import HeadMasker
    m = HeadMasker(tiny_model, cfg, seed=3)
    m.install()
    try:
        m.activate()
        with m.unmasked():
            assert m.masking is False
        assert m.masking is True

        with pytest.raises(RuntimeError):
            with m.unmasked():
                raise RuntimeError("boom")
        assert m.masking is True, "state not restored after an exception"

        m.remove_mask()
        with m.unmasked():
            pass
        assert m.masking is False, "unmasked() must not switch masking on"
    finally:
        m.remove()


def test_custom_step_can_read_unperturbed_targets(tiny_model, cfg, loader):
    """The I-JEPA pattern: unmasked target pass, masked prediction pass."""
    gaps = []

    def step(model, batch, optimizer, mask_module):
        with torch.no_grad(), mask_module.unmasked():
            target = model(**batch).logits
        pred = model(**batch).logits                 # masked
        loss = nn.functional.mse_loss(pred, target.detach())
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        gaps.append(loss.item())
        return loss.item()

    _make(tiny_model, cfg, loader, step).train(num_epochs=2)

    # If the mask were a no-op the two passes would be identical and every
    # loss would be exactly zero. Some must not be.
    assert any(g > 0 for g in gaps), "masked and unmasked passes were identical"


# ------------------------------------------------------- backward compatibility
def test_backward_compat_no_callback(tiny_model, cfg, loader):
    """train_step=None keeps the built-in cross-entropy loop."""
    trainer = _make(tiny_model, cfg, loader)
    assert trainer.train_step is None

    hist = trainer.train(num_epochs=2)
    assert len(hist["losses"]) == 2
    assert all(isinstance(x, float) and x > 0 for x in hist["losses"])
    assert hist["masker_stats"]["pruned_heads"] > 0


def test_backward_compat_positional_signature(tiny_model, cfg, loader):
    """The v0.5.0 constructor call still works unchanged (train_step is last)."""
    opt = SGD(tiny_model.parameters(), lr=1e-3)
    trainer = SALTrainer(tiny_model, cfg, opt, loader, None, 42, 1, 1.0)
    hist = trainer.train(num_epochs=1)
    assert len(hist["losses"]) == 1


def test_default_and_custom_paths_agree(tiny_model, cfg, loader):
    """A callback replicating the default loop reaches the same loss."""
    TinyTransformer = type(tiny_model)

    def run(train_step):
        torch.manual_seed(42)
        model = TinyTransformer()
        return SALTrainer(model, cfg, SGD(model.parameters(), lr=1e-3), loader,
                          seed=7, train_step=train_step).train(num_epochs=2)

    def equivalent(model, batch, optimizer, mask_module):
        loss = model(**batch).loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    assert run(None)["losses"] == pytest.approx(run(equivalent)["losses"], rel=1e-5)


def test_hooks_removed_when_callback_raises(tiny_model, cfg, loader):
    """A failing callback must not leave hooks on the model."""
    def boom(model, batch, optimizer, mask_module):
        raise ValueError("callback exploded")

    trainer = _make(tiny_model, cfg, loader, boom)
    with pytest.raises(ValueError, match="callback exploded"):
        trainer.train(num_epochs=1)
    assert trainer.masker._hooks == []


# ----------------------------------------------------------- non-dict batches
def test_custom_step_with_tuple_batch(tiny_model, cfg):
    """Vision loaders yield (images, labels) tuples; tensors still hit the device."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (8, 16))
    dl = DataLoader(torch.utils.data.TensorDataset(ids, ids), batch_size=2)
    seen = []

    def step(model, batch, optimizer, mask_module):
        assert isinstance(batch, (list, tuple)) and len(batch) == 2
        seen.append(type(batch[0]))
        loss = model(input_ids=batch[0], labels=batch[1]).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return loss.item()

    _make(tiny_model, cfg, dl, step).train(num_epochs=1)
    assert seen and all(t is torch.Tensor for t in seen)
