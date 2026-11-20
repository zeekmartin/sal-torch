"""DropHead, callable from the package, against the prototype that measured it.

The equivalence test is the point of this file: ``mask_mode="drophead"`` must
reproduce the masker the published DropHead cells were trained with, step by
step and head for head, at identical seeds. The reference is the real one,
``scripts/run_sal_qat_prototype.py:ResampleMasker``, imported and stepped
beside the new path rather than reimplemented here.
"""
import sys
from pathlib import Path

import pytest
import torch

from sal.config import SALConfig
from sal.masker import HeadMasker

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _config(**kw):
    base = dict(num_layers=4, num_heads_per_layer=8,
                attention_pattern="transformer.h.{}.attn",
                prune_fraction=0.50)
    base.update(kw)
    return SALConfig(**base)


def _masks(masker) -> list:
    inner = getattr(masker, "inner", masker)
    return [inner._masks[i].clone() for i in sorted(inner._masks)]


def _pruned(masker) -> set:
    inner = getattr(masker, "inner", masker)
    return {(li, hi) for li in sorted(inner._masks)
            for hi in (inner._masks[li] == 0.0).nonzero(as_tuple=True)[0].tolist()}


# -------------------------------------------------------------- the two modes
def test_the_default_is_still_sal():
    assert SALConfig(num_layers=4, num_heads_per_layer=8).mask_mode == "accumulate"


def test_an_unknown_mask_mode_raises():
    with pytest.raises(ValueError, match="mask_mode must be"):
        _config(mask_mode="dropout")


def test_accumulate_holds_what_it_pruned(tiny_model):
    m = HeadMasker(tiny_model, _config(mask_mode="accumulate"), seed=0)
    m.install()
    try:
        seen = []
        for step in range(20):
            m.step(step, 20)
            seen.append(_pruned(m))
        # Every set contains the one before it: nothing is ever restored.
        for before, after in zip(seen, seen[1:]):
            assert before <= after
        assert len(seen[-1]) == m.config.num_heads_to_prune
        assert m.stats["mask_mode"] == "accumulate"
        assert m.stats["resample_events"] == 0
    finally:
        m.remove()


def test_drophead_resamples_and_holds_nothing(tiny_model):
    m = HeadMasker(tiny_model, _config(mask_mode="drophead"), seed=0)
    m.install()
    try:
        quota = m.config.num_heads_to_prune
        sets = []
        for step in range(20):
            m.step(step, 20)
            sets.append(_pruned(m))
            # The full quota from step 0, with no ramp and no window.
            assert len(sets[-1]) == quota, f"step {step}"
        assert len({frozenset(s) for s in sets}) > 1, "the set must change"
        # It is not a growing set: each one drops heads the last one had.
        assert any(not (a <= b) for a, b in zip(sets, sets[1:]))
        assert m.stats["mask_mode"] == "drophead"
        assert m.stats["resample_events"] == 20
    finally:
        m.remove()


def test_drophead_ignores_the_window(tiny_model):
    """prune_start_ratio would hold SAL off for the first tenth of the run."""
    m = HeadMasker(tiny_model, _config(mask_mode="drophead",
                                       prune_start_ratio=0.5), seed=0)
    m.install()
    try:
        m.step(0, 100)
        assert len(_pruned(m)) == m.config.num_heads_to_prune
        assert m.masking is True
    finally:
        m.remove()

    sal = HeadMasker(tiny_model, _config(prune_start_ratio=0.5), seed=0)
    sal.install()
    try:
        sal.step(0, 100)
        assert _pruned(sal) == set(), "SAL waits for its window"
    finally:
        sal.remove()


def test_drophead_needs_the_hooks_installed(tiny_model):
    m = HeadMasker(tiny_model, _config(mask_mode="drophead"), seed=0)
    with pytest.raises(RuntimeError, match="Not installed"):
        m.step(0, 10)


# ------------------------------------------------- equivalence with the proto
@pytest.fixture
def prototype():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import run_sal_qat_prototype as P
        yield P
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.mark.parametrize("seed,steps,fraction", [
    (0, 25, 0.50),
    (42, 40, 0.25),
    (123, 12, 0.75),
])
def test_drophead_matches_the_prototype_step_for_step(tiny_model, prototype,
                                                      seed, steps, fraction):
    """Identical seeds, identical masks, at every step.

    Both draw from their own RNG, so this also pins the order of operations
    inside the resample: clearing the masks before drawing, and drawing once
    for the whole quota. A different order gives a different set from the same
    seed and the comparison with the published cells would be lost.
    """
    import copy

    ours = HeadMasker(copy.deepcopy(tiny_model),
                      _config(mask_mode="drophead", prune_fraction=fraction),
                      seed=seed)
    theirs = prototype.ResampleMasker(copy.deepcopy(tiny_model),
                                      _config(prune_fraction=fraction),
                                      seed=seed, mode="drophead")
    ours.install()
    theirs.install()
    try:
        for step in range(steps):
            ours.step(step, steps)
            theirs.step(step, steps)
            a, b = _masks(ours), _masks(theirs)
            assert all(torch.equal(x, y) for x, y in zip(a, b)), f"step {step}"
            assert _pruned(ours) == _pruned(theirs)
        assert ours.stats["pruned_heads"] == theirs.stats["pruned_heads"]
    finally:
        ours.remove()
        theirs.remove()


def test_the_mechanism_is_the_same_hook(tiny_model, prototype):
    """Same masks must mean the same logits, or the mask is not being applied."""
    import copy

    batch = {"input_ids": torch.randint(0, 100, (2, 8))}
    ours = HeadMasker(copy.deepcopy(tiny_model), _config(mask_mode="drophead"),
                      seed=7)
    theirs = prototype.ResampleMasker(copy.deepcopy(tiny_model), _config(),
                                      seed=7, mode="drophead")
    ours.install()
    theirs.install()
    try:
        ours.step(3, 10)
        theirs.step(3, 10)
        with torch.no_grad():
            a = ours.model(**batch).logits
            b = theirs.inner.model(**batch).logits
        assert torch.allclose(a, b, atol=1e-6)

        # And the mask is doing something: unmasked logits differ.
        with ours.unmasked(), torch.no_grad():
            c = ours.model(**batch).logits
        assert not torch.allclose(a, c, atol=1e-6)
    finally:
        ours.remove()
        theirs.remove()


# ------------------------------------------------------------- through a config
def test_auto_does_not_change_the_default(tiny_model):
    cfg = SALConfig.auto(tiny_model)
    assert cfg.mask_mode == "accumulate", (
        "the mode is an explicit choice: the two are different methods with "
        "different scopes, and auto() must not pick one")


def test_the_trainer_passes_the_mode_through(tiny_model):
    from sal.trainer import SALTrainer

    data = [{"input_ids": torch.randint(0, 100, (2, 8)),
             "labels": torch.randint(0, 100, (2, 8))} for _ in range(2)]
    cfg = _config(mask_mode="drophead")
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    trainer = SALTrainer(tiny_model, cfg, opt, data, seed=0)
    trainer.train(num_epochs=1)
    assert trainer.masker.config.mask_mode == "drophead"
    assert trainer.masker.stats["resample_events"] > 0
