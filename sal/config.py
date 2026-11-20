"""SAL configuration with auto-detection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SALConfig:
    num_layers: int = 0
    num_heads_per_layer: int = 0
    prune_fraction: float = 0.50
    prune_start_ratio: float = 0.10
    prune_end_ratio: float = 0.80
    prune_interval: int = 1
    schedule: str = "random"
    mask_mode: str = "accumulate"
    uniform_per_layer: bool = True
    fi_interval: int = 0
    attention_pattern: Optional[str] = None
    head_dim: Optional[int] = None
    _total_heads: int = field(init=False, repr=False, default=0)

    #: ``"accumulate"`` is SAL: heads are pruned progressively and stay
    #: pruned, so the adaptation is to one head set, fixed before training.
    #: ``"drophead"`` resamples the full quota every step and holds nothing,
    #: so the resilience it trains applies to any set of that size.
    #:
    #: Which to use is not a preference. **If a post-pruning fine-tune is
    #: available, neither pays**: at a fixed head set SAL's weights are worth
    #: +0.05pp and DropHead's +0.07pp after recovery, against the recovery
    #: itself being worth up to 68pp, and on BERT-base across four GLUE tasks
    #: magnitude plus 2 epochs of recovery beat DropHead on 8 of 8 cells by
    #: 0.42 to 7.03pp over 3 seeds. If no fine-tune is possible, DropHead,
    #: with no exception: on a 631M self-supervised encoder it is +0.67pp
    #: above SAL on 3 of 3 seeds even on SAL's own trained-for head set, which
    #: was the last cell where SAL could have been the right choice.
    #: Public data archive, DOI 10.5281/zenodo.24391200.
    MASK_MODES = ("accumulate", "drophead")

    def __post_init__(self):
        if self.schedule not in ("random", "progressive", "burst"):
            raise ValueError(f"schedule must be 'random'/'progressive'/'burst', got '{self.schedule}'")
        if self.mask_mode not in self.MASK_MODES:
            raise ValueError(f"mask_mode must be one of {self.MASK_MODES}, "
                             f"got '{self.mask_mode}'")
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
        n = max(1, int(self._total_heads * self.prune_fraction))
        if self.uniform_per_layer and self.num_layers > 0:
            # An equal quota per layer is only achievable when the target is a
            # multiple of the layer count: 0.33 of a 12x12 model is 43 heads,
            # which cannot be split evenly however the draw is constrained. Snap
            # to the nearest reachable count so the trained set really is a
            # slice_heads() argument. Read `effective_prune_fraction` for what
            # was actually requested of the model.
            n = max(self.num_layers,
                    min(self._total_heads,
                        round(n / self.num_layers) * self.num_layers))
        return n

    @property
    def effective_prune_fraction(self) -> float:
        """The fraction actually pruned, which ``uniform_per_layer`` may round."""
        if self._total_heads == 0:
            return 0.0
        return self.num_heads_to_prune / self._total_heads

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
