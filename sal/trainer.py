"""Standalone SAL training loop (no HF dependency)."""
from __future__ import annotations
import logging
from typing import Optional
import torch, torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sal.config import SALConfig
from sal.masker import HeadMasker

logger = logging.getLogger(__name__)

class SALTrainer:
    def __init__(self, model: nn.Module, config: SALConfig, optimizer: Optimizer,
                 train_dataloader: DataLoader, scheduler=None, seed: Optional[int] = None,
                 gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0):
        self.model = model; self.config = config; self.optimizer = optimizer
        self.train_dl = train_dataloader; self.scheduler = scheduler
        self.grad_accum = gradient_accumulation_steps; self.max_grad_norm = max_grad_norm
        self.device = next(model.parameters()).device
        self.masker = HeadMasker(model, config, seed=seed)

    def train(self, num_epochs: int, log_interval: int = 50) -> dict:
        total_steps = (len(self.train_dl) * num_epochs) // self.grad_accum
        self.masker.install()
        self.model.train()
        global_step = 0; losses = []
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0; steps = 0
                for bi, batch in enumerate(self.train_dl):
                    if isinstance(batch, dict):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    self.masker.step(global_step, total_steps)
                    out = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                    loss = out.loss if hasattr(out, 'loss') else out
                    (loss / self.grad_accum).backward()
                    epoch_loss += loss.item(); steps += 1
                    if (bi + 1) % self.grad_accum == 0:
                        if self.max_grad_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        if self.scheduler: self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                losses.append(epoch_loss / max(steps, 1))
        finally:
            self.masker.remove()
        return {"losses": losses, "total_steps": global_step, "masker_stats": self.masker.stats}
