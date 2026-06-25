"""Tests for HeadMasker."""
import pytest, torch
from sal.masker import HeadMasker

class TestHeadMasker:
    def test_install_remove(self, tiny_model, tiny_config):
        m = HeadMasker(tiny_model, tiny_config)
        m.install()
        assert len(m._hooks) == 4
        m.remove()
        assert len(m._hooks) == 0

    def test_double_install(self, tiny_model, tiny_config):
        m = HeadMasker(tiny_model, tiny_config)
        m.install()
        with pytest.raises(RuntimeError):
            m.install()
        m.remove()

    def test_mask_application(self, tiny_model, tiny_config):
        m = HeadMasker(tiny_model, tiny_config, seed=42)
        m.install(); m.activate()
        pruned = sum((mask == 0).sum().item() for mask in m._masks.values())
        assert pruned == tiny_config.num_heads_to_prune
        m.remove()

    def test_forward_changes_output(self, tiny_model, tiny_config):
        ids = torch.randint(0, 100, (2, 16))
        tiny_model.eval()
        with torch.no_grad():
            clean = tiny_model(input_ids=ids).logits.clone()
        m = HeadMasker(tiny_model, tiny_config, seed=42)
        m.install(); m.activate()
        with torch.no_grad():
            masked = tiny_model(input_ids=ids).logits.clone()
        m.remove()
        assert not torch.allclose(clean, masked, atol=1e-6)

    def test_deactivate_restores(self, tiny_model, tiny_config):
        m = HeadMasker(tiny_model, tiny_config, seed=42)
        m.install(); m.activate(); m.deactivate()
        for mask in m._masks.values():
            assert torch.all(mask == 1.0)
        m.remove()

    def test_window_management(self, tiny_model, tiny_config):
        m = HeadMasker(tiny_model, tiny_config, seed=42)
        m.install()
        m.step(5, 100); assert not m._active   # 5% < 10%: before window
        m.step(50, 100); assert m._active      # 50% in [10%, 80%]: pruning
        m.step(85, 100); assert m._active      # 85% > 80%: pruned set held (not restored)
        m.remove()

    def test_progressive_accumulation(self, tiny_model, tiny_config):
        # Pruned heads accumulate monotonically and reach the full target by the
        # end of the prune window (the validated "progressive damage" mechanism).
        m = HeadMasker(tiny_model, tiny_config, seed=42)
        m.install()
        target = tiny_config.num_heads_to_prune

        def pruned():
            return sum((mask == 0).sum().item() for mask in m._masks.values())

        m.step(10, 100); early = pruned()    # window start: ~0
        m.step(45, 100); mid = pruned()      # mid window
        m.step(80, 100); end = pruned()      # window end: full target
        assert early <= mid <= end           # accumulation never shrinks
        assert end == target
        m.remove()

    def test_seed_reproducibility(self, tiny_model, tiny_config):
        m1 = HeadMasker(tiny_model, tiny_config, seed=123)
        m1.install(); m1.activate()
        masks1 = {k: v.clone() for k,v in m1._masks.items()}
        m1.remove()
        m2 = HeadMasker(tiny_model, tiny_config, seed=123)
        m2.install(); m2.activate()
        for k in masks1:
            assert torch.equal(masks1[k], m2._masks[k])
        m2.remove()

    def test_stats(self, tiny_model, tiny_config):
        m = HeadMasker(tiny_model, tiny_config, seed=42)
        m.install()
        assert m.stats["total_heads"] == 32
        assert not m.stats["active"]
        m.activate()
        assert m.stats["active"]
        assert m.stats["pruned_heads"] == tiny_config.num_heads_to_prune
        m.remove()
