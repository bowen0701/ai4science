from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ai4science.utils.logging import setup_logger

logger = setup_logger("DiffusionLearner")


@dataclass
class DiffusionBatch:
    noisy_inputs: torch.Tensor
    target_noise: torch.Tensor
    conditioning: torch.Tensor | None = None


class DiffusionLearner:
    """Minimal diffusion learner with explicit train and sampling steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or torch.nn.MSELoss()

    def train_step(self, batch: DiffusionBatch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(batch.noisy_inputs, conditioning=batch.conditioning)
        loss = self.loss_fn(prediction, batch.target_noise)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def sample(
        self,
        latent: torch.Tensor,
        num_steps: int,
        step_fn: Callable[[torch.nn.Module, torch.Tensor, int, torch.Tensor | None], torch.Tensor],
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        state = latent
        for step_idx in range(num_steps):
            state = step_fn(self.model, state, step_idx, conditioning)
        return state
