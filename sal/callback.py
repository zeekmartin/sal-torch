"""SAL integration with HuggingFace Trainer."""
from __future__ import annotations
import logging
from typing import Optional
from sal.config import SALConfig
from sal.masker import HeadMasker

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
    HAS_TF = True
except ImportError:
    HAS_TF = False
    class TrainerCallback: pass

class SALCallback(TrainerCallback):
    def __init__(self, config: SALConfig, seed: Optional[int] = None):
        if not HAS_TF:
            raise ImportError("SALCallback requires transformers. pip install sal-torch[hf]")
        self.config = config
        self.seed = seed
        self.masker: Optional[HeadMasker] = None
        self._total_steps = 0

    def on_train_begin(self, args, state, control, model=None, **kw):
        if model is None:
            raise RuntimeError("SALCallback needs model")
        self.config.validate_for_model(model)
        self.masker = HeadMasker(model, self.config, seed=self.seed)
        self.masker.install()
        self._total_steps = state.max_steps

    def on_step_begin(self, args, state, control, **kw):
        if self.masker:
            self.masker.step(state.global_step, self._total_steps)

    def on_log(self, args, state, control, logs=None, **kw):
        if self.masker and logs:
            s = self.masker.stats
            logs["sal/active_heads"] = s["active_heads"]
            logs["sal/pruned_heads"] = s["pruned_heads"]
            logs["sal/prune_events"] = s["prune_events"]

    def on_train_end(self, args, state, control, **kw):
        if self.masker:
            self.masker.remove()
            self.masker = None
