"""SAL configuration with auto-detection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SALConfig:
    num_layers: int = 0
    num_heads_per_layer: int = 0
    prune_fraction: float = 0.33
    prune_start_ratio: float = 0.10
    prune_end_ratio: float = 0.80
    prune_interval: int = 1
    schedule: str = "random"
    fi_interval: int = 0
    attention_pattern: Optional[str] = None
    head_dim: Optional[int] = None
    _total_heads: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if self.schedule not in ("random", "progressive", "burst"):
            raise ValueError(f"schedule must be 'random'/'progressive'/'burst', got '{self.schedule}'")
        if not 0.0 < self.prune_fraction < 1.0:
            raise ValueError(f"prune_fraction must be in (0,1), got {self.prune_fraction}")
        if self.prune_start_ratio >= self.prune_end_ratio:
            raise ValueError("prune_start_ratio must be < prune_end_ratio")
        self._total_heads = self.num_layers * self.num_heads_per_layer

    @property
    def total_heads(self) -> int:
        return self._total_heads

    @property
    def num_heads_to_prune(self) -> int:
        return max(1, int(self._total_heads * self.prune_fraction))

    @classmethod
    def auto(cls, model, **overrides) -> SALConfig:
        from sal.arch_support import detect_architecture
        info = detect_architecture(model)
        defaults = dict(num_layers=info.num_layers, num_heads_per_layer=info.num_heads,
                       attention_pattern=info.attention_pattern, head_dim=info.head_dim)
        defaults.update(overrides)
        return cls(**defaults)

    def validate_for_model(self, model) -> None:
        from sal.arch_support import detect_architecture, SALArchitectureError
        try:
            info = detect_architecture(model)
        except SALArchitectureError:
            return
        if info.num_layers != self.num_layers:
            raise ValueError(f"Config has {self.num_layers} layers but model has {info.num_layers}")
        if info.num_heads != self.num_heads_per_layer:
            raise ValueError(f"Config has {self.num_heads_per_layer} heads but model has {info.num_heads}")
