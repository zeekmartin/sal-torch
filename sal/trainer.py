"""Standalone SAL training loop (no HF dependency).

By default the loop is supervised: each batch is fed to the model, the model's
own ``.loss`` (cross-entropy, for the task heads this package targets) is
backpropagated, and gradients are clipped and accumulated for you.

That default is only *one* way to compute a loss. The SAL mechanism itself —
progressively zeroing random attention heads via :class:`~sal.masker.HeadMasker`
— knows nothing about the objective. Pass ``train_step=`` and the loop hands you
the model, the batch, the optimizer and the masker, and you own the whole step:

    def my_train_step(model, batch, optimizer, mask_module):
        mask_module.apply_mask()
        loss = my_objective(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mask_module.remove_mask()
        return loss.item()

    trainer = SALTrainer(model, config, optimizer, dl, train_step=my_train_step)

This is what makes SAL usable for self-supervised objectives (I-JEPA, V-JEPA,
MAE, DINO) where the loss lives in the training script rather than in the model
head. See ``examples/jepa_sal.py``.

**What the loop still owns in custom mode:** the prune schedule. ``step()`` is
called on the masker before every ``train_step`` call, so heads accumulate
exactly as they do in supervised mode, and masking is left *on* when your
callback is entered. **What your callback owns:** the forward pass, the loss,
``backward()``, gradient clipping, ``optimizer.step()``, ``zero_grad()``, and
the scheduler. ``gradient_accumulation_steps`` and ``max_grad_norm`` are not
applied for you — a custom objective may accumulate differently.
"""
from __future__ import annotations
import logging
from typing import Callable, Optional
import torch, torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sal.config import SALConfig
from sal.masker import HeadMasker

logger = logging.getLogger(__name__)

# train_step(model, batch, optimizer, mask_module) -> loss (float or 0-d Tensor)
TrainStep = Callable[[nn.Module, object, Optimizer, HeadMasker], object]


class SALTrainer:
    def __init__(self, model: nn.Module, config: SALConfig, optimizer: Optimizer,
                 train_dataloader: DataLoader, scheduler=None, seed: Optional[int] = None,
                 gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0,
                 train_step: Optional[TrainStep] = None):
        if train_step is not None and not callable(train_step):
            raise TypeError(
                "train_step must be callable with signature "
                "(model, batch, optimizer, mask_module) -> loss, "
                f"got {type(train_step).__name__}.")
        self.model = model; self.config = config; self.optimizer = optimizer
        self.train_dl = train_dataloader; self.scheduler = scheduler
        self.grad_accum = gradient_accumulation_steps; self.max_grad_norm = max_grad_norm
        self.train_step = train_step
        self.device = next(model.parameters()).device
        self.masker = HeadMasker(model, config, seed=seed)

    def train(self, num_epochs: int, log_interval: int = 50) -> dict:
        if self.train_step is not None:
            return self._train_custom(num_epochs)
        return self._train_supervised(num_epochs)

    # ------------------------------------------------------------- supervised
    def _train_supervised(self, num_epochs: int) -> dict:
        total_steps = (len(self.train_dl) * num_epochs) // self.grad_accum
        self.masker.install()
        self.model.train()
        global_step = 0; losses = []
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0; steps = 0
                for bi, batch in enumerate(self.train_dl):
                    batch = self._to_device(batch)
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
            stats = self.masker.stats
        finally:
            self.masker.remove()
        return {"losses": losses, "total_steps": global_step, "masker_stats": stats}

    # ----------------------------------------------------------------- custom
    def _train_custom(self, num_epochs: int) -> dict:
        """Loss-agnostic loop: the callback owns everything but the schedule.

        One optimizer step per batch is assumed for scheduling purposes (the
        callback is what actually steps), so ``total_steps`` is the batch count.
        """
        total_steps = len(self.train_dl) * num_epochs
        self.masker.install()
        self.model.train()
        global_step = 0; losses = []
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0; steps = 0
                for batch in self.train_dl:
                    batch = self._to_device(batch)
                    # Advance the prune schedule and leave masking on, so a
                    # callback that never touches the masker is still SAL-trained.
                    self.masker.step(global_step, total_steps)
                    loss = self.train_step(self.model, batch, self.optimizer, self.masker)
                    epoch_loss += self._as_float(loss); steps += 1
                    if self.scheduler: self.scheduler.step()
                    global_step += 1
                losses.append(epoch_loss / max(steps, 1))
            stats = self.masker.stats
        finally:
            self.masker.remove()
        return {"losses": losses, "total_steps": global_step, "masker_stats": stats}

    # -------------------------------------------------------------- internals
    def _to_device(self, batch):
        """Move tensors in a dict / list / tuple / tensor batch onto the model."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for v in batch)
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    @staticmethod
    def _as_float(loss) -> float:
        if loss is None:
            return 0.0
        if isinstance(loss, torch.Tensor):
            return float(loss.detach().item())
        try:
            return float(loss)
        except (TypeError, ValueError):
            raise TypeError(
                "train_step must return a float or a scalar Tensor (the step loss), "
                f"got {type(loss).__name__}.")
