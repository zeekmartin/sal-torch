"""Head masking mechanism for SAL.

The validated mechanism (see reference research, RANDOM is the default method):

  * Heads are zeroed via a forward **pre-hook** on the attention *output
    projection*. The hook zeros the per-head slices of the projection's
    INPUT — the concatenated per-head attention outputs, the only place where
    feature dimensions map cleanly onto heads. Zeroing the projection output
    would zero arbitrary mixed features, not heads.
  * Pruned heads **accumulate**. During the prune window, randomly chosen heads
    are deactivated progressively and stay deactivated. This is the validated
    "progressive structural damage" mechanism — the model adapts to operate
    without the removed heads.
  * After the prune window the pruned set is **held** through the end of
    training (not restored), so the adaptation is baked into the weights.

`schedule` controls how the pruned-head count grows over the window:
  * ``"random"``       — default. Count ramps with window progress; additional
                         random heads are accumulated to track the ramp.
  * ``"progressive"``  — one additional random head every ``prune_interval``
                         steps (fixed cadence), capped at the target.
  * ``"burst"``        — the full target is deactivated at once when the window
                         opens.

All three accumulate and use random selection (the only validated, zero-overhead
selection method).
"""
from __future__ import annotations
import logging, random
from typing import Optional
import torch, torch.nn as nn
from sal.config import SALConfig
from sal import arch_support

logger = logging.getLogger(__name__)


class HeadMasker:
    def __init__(self, model: nn.Module, config: SALConfig, seed: Optional[int] = None):
        self.model = model
        self.config = config
        self.rng = random.Random(seed)
        self._hooks: list = []
        self._masks: dict[int, torch.Tensor] = {}
        self._attention_modules: list[nn.Module] = []
        self._active = False
        self._step_count = 0
        self._prune_events = 0

    # ------------------------------------------------------------------ setup
    def install(self):
        if self._hooks:
            raise RuntimeError("HeadMasker already installed. Call remove() first.")
        self._attention_modules = arch_support.get_attention_modules(
            self.model, self.config.attention_pattern)
        if len(self._attention_modules) != self.config.num_layers:
            raise ValueError(
                f"Found {len(self._attention_modules)} attention modules, "
                f"config says {self.config.num_layers}. "
                f"Pass attention_pattern=... to SALConfig if auto-detection missed them.")
        device = next(self.model.parameters()).device
        for layer_idx, attn_mod in enumerate(self._attention_modules):
            out_proj = arch_support.get_output_projection(attn_mod)
            if out_proj is None:
                raise ValueError(
                    f"No attention output projection found at layer {layer_idx} "
                    f"({type(attn_mod).__name__}).")
            hook = out_proj.register_forward_pre_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)
            self._masks[layer_idx] = torch.ones(self.config.num_heads_per_layer, device=device)
        logger.info(
            f"SAL HeadMasker installed: {len(self._attention_modules)} layers, "
            f"{self.config.total_heads} heads")

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear(); self._masks.clear(); self._attention_modules.clear()
        self._active = False

    # ------------------------------------------------------------- activation
    def activate(self):
        """Deactivate the full target set of heads immediately (burst).

        Used for standalone/manual control and as the burst schedule. The
        progressive accumulation during training is driven by ``step()``.
        """
        if not self._hooks:
            raise RuntimeError("Not installed")
        self._active = True
        for m in self._masks.values():
            m.fill_(1.0)
        self._prune_to_count(self.config.num_heads_to_prune)

    def deactivate(self):
        self._active = False
        for m in self._masks.values():
            m.fill_(1.0)

    def step(self, global_step: int, total_steps: int):
        self._step_count = global_step
        progress = global_step / max(total_steps, 1)
        start, end = self.config.prune_start_ratio, self.config.prune_end_ratio

        if progress < start:
            if self._active:
                self.deactivate()
            return

        # In the prune window or past it: masking is active and the pruned set
        # is held (never restored) through the end of training.
        self._active = True

        interval = max(1, self.config.prune_interval)
        if global_step % interval != 0:
            return

        self._prune_to_count(self._scheduled_count(global_step, total_steps))

    # -------------------------------------------------------------- internals
    def _scheduled_count(self, global_step: int, total_steps: int) -> int:
        """Target number of pruned heads at this step, per schedule."""
        full = self.config.num_heads_to_prune
        if self.config.schedule == "burst":
            return full

        start, end = self.config.prune_start_ratio, self.config.prune_end_ratio
        if self.config.schedule == "progressive":
            interval = max(1, self.config.prune_interval)
            start_step = int(start * total_steps)
            ticks = (global_step - start_step) // interval + 1
            return max(0, min(full, ticks))

        # "random" (default): ramp the count with window progress.
        progress = global_step / max(total_steps, 1)
        span = max(end - start, 1e-9)
        w = min(1.0, max(0.0, (progress - start) / span))
        return min(full, int(round(full * w)))

    def _prune_to_count(self, target: int):
        """Accumulate randomly chosen heads until ``target`` are pruned."""
        current = self.config.total_heads - self._active_head_count()
        need = target - current
        if need <= 0:
            return
        self._deactivate_random_heads(need)
        self._prune_events += 1

    def _deactivate_random_heads(self, n: int):
        active = [(l, h)
                  for l in range(self.config.num_layers)
                  for h in range(self.config.num_heads_per_layer)
                  if self._masks[l][h].item() == 1.0]
        for li, hi in self.rng.sample(active, min(n, len(active))):
            self._masks[li][hi] = 0.0

    def _active_head_count(self) -> int:
        return int(sum(m.sum().item() for m in self._masks.values()))

    def _make_hook(self, layer_idx: int):
        nh = self.config.num_heads_per_layer

        def pre_hook(module, inputs):
            if not self._active:
                return None
            mask = self._masks.get(layer_idx)
            if mask is None or bool(mask.all()):
                return None
            x = inputs[0]
            bs, sl, hidden = x.shape
            head_dim = hidden // nh
            # Input to the output projection is the concatenation of per-head
            # attention outputs: dims map onto heads here. Zero whole-head
            # slices, preserving the autograd graph for the surviving heads.
            x = (x.view(bs, sl, nh, head_dim) * mask.view(1, 1, nh, 1)).reshape(bs, sl, hidden)
            return (x,) + tuple(inputs[1:])

        return pre_hook

    @property
    def stats(self) -> dict:
        active = self._active_head_count() if self._masks else 0
        return {"active": self._active, "total_heads": self.config.total_heads,
                "active_heads": active, "pruned_heads": self.config.total_heads - active,
                "prune_events": self._prune_events, "prune_fraction": self.config.prune_fraction}
