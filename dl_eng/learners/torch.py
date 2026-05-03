import torch
from typing import Any, Callable
from dl_eng.interfaces.learner_interface import LearnerInterface
from dl_eng.infra.logger import setup_logger

logger = setup_logger("TorchLearner")

class TorchLearner(LearnerInterface):
    """Generic learner for PyTorch-based models.

    Usage:
        learner = TorchLearner(model, optimizer, loss_fn)
        metrics = learner.train_step((x, y))
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: Callable
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        self.model.train()
        x, y = batch
        
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        self.model.eval()
        x, y = batch
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
        return {"loss": loss.item()}
