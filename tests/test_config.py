"""Tests for SALConfig."""
import pytest
from sal.config import SALConfig

class TestSALConfig:
    def test_basic(self):
        c = SALConfig(num_layers=12, num_heads_per_layer=12)
        assert c.total_heads == 144
        assert c.num_heads_to_prune == 47

    def test_defaults(self):
        c = SALConfig(num_layers=4, num_heads_per_layer=8)
        assert c.prune_fraction == 0.33
        assert c.schedule == "random"

    def test_invalid_schedule(self):
        with pytest.raises(ValueError):
            SALConfig(num_layers=4, num_heads_per_layer=8, schedule="bad")

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            SALConfig(num_layers=4, num_heads_per_layer=8, prune_fraction=1.5)

    def test_auto(self, tiny_model):
        c = SALConfig.auto(tiny_model)
        assert c.num_layers == 4
        assert c.num_heads_per_layer == 8

    def test_validate(self, tiny_model, tiny_config):
        tiny_config.validate_for_model(tiny_model)  # should not raise

    def test_validate_mismatch(self, tiny_model):
        wrong = SALConfig(num_layers=12, num_heads_per_layer=8)
        with pytest.raises(ValueError):
            wrong.validate_for_model(tiny_model)
