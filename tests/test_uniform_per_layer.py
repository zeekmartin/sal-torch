"""SALConfig(uniform_per_layer=True) — an equal pruning quota in every layer.

The default draw takes heads from the whole model, so the trained set is not
uniform per layer and ``slice_heads()`` cannot express it: the accuracy gain
measured in exp12 is reachable only by masking, which makes nothing smaller or
faster. This option constrains the draw so the set SAL trains for is exactly a
``slice_heads()`` argument.
"""
import pytest
import torch

from sal.config import SALConfig
from sal.masker import HeadMasker

PATTERN = "transformer.h.{}.attn"


def _config(**kw):
    base = dict(num_layers=4, num_heads_per_layer=8, attention_pattern=PATTERN,
                prune_fraction=0.5)
    base.update(kw)
    return SALConfig(**base)


def _per_layer(masker, num_layers):
    return [int((masker._masks[l] == 0.0).sum().item()) for l in range(num_layers)]


def _arm(masker):
    """Switch masking on without pruning anything.

    ``activate()`` is the burst schedule: it immediately prunes the whole target
    set, so calling it first makes every subsequent count 16 heads too high.
    These tests drive ``_deactivate_random_heads`` directly.
    """
    masker._active = True
    for m in masker._masks.values():
        m.fill_(1.0)


def test_default_stays_global_and_unconstrained(tiny_model):
    """Cardinal rule: the shipped default is unchanged."""
    assert _config().uniform_per_layer is False


def test_uniform_draw_gives_every_layer_the_same_count(tiny_model):
    cfg = _config(uniform_per_layer=True)
    m = HeadMasker(tiny_model, cfg, seed=0)
    m.install()
    try:
        _arm(m)
        m._deactivate_random_heads(16)          # 4 layers x 4 heads
        assert _per_layer(m, 4) == [4, 4, 4, 4]
    finally:
        m.remove()


@pytest.mark.parametrize("n", [1, 3, 5, 7, 11, 13, 17, 23, 31])
def test_uniform_draw_never_differs_by_more_than_one(tiny_model, n):
    """Holds at every intermediate count, not just at multiples of num_layers."""
    cfg = _config(uniform_per_layer=True)
    m = HeadMasker(tiny_model, cfg, seed=n)
    m.install()
    try:
        _arm(m)
        m._deactivate_random_heads(n)
        counts = _per_layer(m, 4)
        assert sum(counts) == n
        assert max(counts) - min(counts) <= 1
    finally:
        m.remove()


def test_accumulating_in_steps_stays_balanced(tiny_model):
    """The quota is kept along the schedule, not restored at the end."""
    cfg = _config(uniform_per_layer=True)
    m = HeadMasker(tiny_model, cfg, seed=7)
    m.install()
    try:
        _arm(m)
        for _ in range(8):
            m._deactivate_random_heads(2)
            counts = _per_layer(m, 4)
            assert max(counts) - min(counts) <= 1
        assert _per_layer(m, 4) == [4, 4, 4, 4]
    finally:
        m.remove()


def test_uniform_set_is_a_valid_slice_heads_argument(tiny_model):
    """The point of the option: pruned_heads() becomes sliceable."""
    cfg = _config(uniform_per_layer=True)
    m = HeadMasker(tiny_model, cfg, seed=3)
    m.install()
    try:
        _arm(m)
        m._deactivate_random_heads(cfg.num_heads_to_prune)
        by_layer = m.pruned_heads_by_layer()
        heads = m.pruned_heads()
    finally:
        m.remove()
    counts = {len(v) for v in by_layer.values()}
    assert len(counts) == 1, f"not uniform: {by_layer}"
    assert len(by_layer) == cfg.num_layers
    assert len(heads) == cfg.num_heads_to_prune


def test_default_draw_is_typically_not_uniform(tiny_model):
    """Contrast: the unconstrained draw is what exp12 measured as unsliceable."""
    cfg = _config()
    seen_nonuniform = False
    for seed in range(10):
        m = HeadMasker(tiny_model, cfg, seed=seed)
        m.install()
        try:
            _arm(m)
            m._deactivate_random_heads(cfg.num_heads_to_prune)
            if len({len(v) for v in m.pruned_heads_by_layer().values()}) > 1:
                seen_nonuniform = True
        finally:
            m.remove()
    assert seen_nonuniform


def test_uniform_draw_still_masks_and_changes_the_output(tiny_model):
    cfg = _config(uniform_per_layer=True)
    x = torch.randint(0, 100, (2, 16))
    tiny_model.eval()
    with torch.no_grad():
        clean = tiny_model(input_ids=x).logits.clone()
    m = HeadMasker(tiny_model, cfg, seed=1)
    m.install()
    try:
        _arm(m)
        m._deactivate_random_heads(16)
        m.apply_mask()
        with torch.no_grad():
            masked = tiny_model(input_ids=x).logits.clone()
    finally:
        m.remove()
    assert (masked - clean).abs().max() > 1e-5
    with torch.no_grad():
        assert torch.allclose(tiny_model(input_ids=x).logits, clean, atol=1e-6)


def test_uniform_draw_is_deterministic_for_a_seed(tiny_model):
    def draw(seed):
        m = HeadMasker(tiny_model, _config(uniform_per_layer=True), seed=seed)
        m.install()
        try:
            _arm(m)
            m._deactivate_random_heads(12)
            return m.pruned_heads()
        finally:
            m.remove()
    assert draw(5) == draw(5)
    assert draw(5) != draw(6)


# ------------------------------------------------- the target must be divisible
def test_target_is_snapped_to_a_multiple_of_the_layer_count():
    """0.33 of a 12x12 model is 43 heads, which no draw can split evenly.

    Without snapping, --uniform-per-layer would silently produce a set that is
    still unsliceable, which is the whole thing the option exists to fix.
    """
    loose = SALConfig(num_layers=12, num_heads_per_layer=12, prune_fraction=0.33)
    assert loose.num_heads_to_prune == 47          # 0.33 * 144, floored

    tight = SALConfig(num_layers=12, num_heads_per_layer=12, prune_fraction=0.33,
                      uniform_per_layer=True)
    assert tight.num_heads_to_prune % 12 == 0
    assert tight.num_heads_to_prune == 48
    assert tight.effective_prune_fraction == pytest.approx(48 / 144)


def test_target_untouched_when_already_divisible():
    cfg = SALConfig(num_layers=12, num_heads_per_layer=12, prune_fraction=0.5,
                    uniform_per_layer=True)
    assert cfg.num_heads_to_prune == 72
    assert cfg.effective_prune_fraction == pytest.approx(0.5)


def test_snapping_never_prunes_everything_or_nothing():
    for frac in (0.01, 0.05, 0.5, 0.9, 0.99):
        cfg = SALConfig(num_layers=12, num_heads_per_layer=12,
                        prune_fraction=frac, uniform_per_layer=True)
        n = cfg.num_heads_to_prune
        assert n % 12 == 0
        assert 12 <= n <= 144


def test_full_target_draw_is_exactly_uniform(tiny_model):
    """The property the experiment rests on: pruning the target leaves an
    equal count in every layer, so slice_heads() accepts it."""
    for frac in (0.25, 0.3, 0.33, 0.5, 0.75):
        cfg = _config(prune_fraction=frac, uniform_per_layer=True)
        m = HeadMasker(tiny_model, cfg, seed=11)
        m.install()
        try:
            _arm(m)
            m._deactivate_random_heads(cfg.num_heads_to_prune)
            counts = _per_layer(m, cfg.num_layers)
            assert len(set(counts)) == 1, f"frac {frac}: {counts}"
        finally:
            m.remove()


def test_full_schedule_ends_uniform_and_on_target(tiny_model):
    """End-to-end over the real ramp, not a single draw.

    The smoke run on LUMI stopped mid-ramp (4 steps) and ended at 67 of 72
    heads, which is uniform to within one. This checks that a schedule with
    enough steps reaches the target exactly, where uniformity is exact.
    """
    cfg = _config(prune_fraction=0.5, uniform_per_layer=True)
    m = HeadMasker(tiny_model, cfg, seed=4)
    m.install()
    try:
        total = 200
        for step in range(total):
            m.step(step, total)
        counts = _per_layer(m, cfg.num_layers)
        assert sum(counts) == cfg.num_heads_to_prune
        assert len(set(counts)) == 1, counts
        heads = m.pruned_heads()
    finally:
        m.remove()
    assert len(heads) == cfg.num_heads_to_prune
