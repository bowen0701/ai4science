from typing import Any, Callable

import numpy as np

from ai4science.utils.interfaces import LearnerProtocol
from ai4science.utils.logging import setup_logger

logger = setup_logger("SupervisedLearner")

class NumPyLearner(LearnerProtocol):
    """Generic learner for NumPy-based models."""

    def __init__(
        self, 
        model: Any, 
        lr: float = 0.01, 
        loss_fn: Callable = None
    ) -> None:
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn or self._default_mse_loss

    def _default_mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.square(y_true - y_pred))

    def train_step(self, batch: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        x, y = batch
        y_pred = self.model.forward(x)
        loss = self.loss_fn(y, y_pred)
        
        # Simple SGD optimization for Linear Regression (needs generalization)
        m = x.shape[0]
        grad_w = 1 / m * np.matmul(x.T, y_pred - y)
        grad_b = np.mean(y_pred - y)
        
        params = self.model.get_params()
        params["w"] -= self.lr * grad_w
        params["b"] -= self.lr * grad_b
        self.model.set_params(params["w"], params["b"])
        
        return {"loss": loss}

    def val_step(self, batch: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        x, y = batch
        y_pred = self.model.forward(x)
        loss = self.loss_fn(y, y_pred)
        return {"loss": loss}
