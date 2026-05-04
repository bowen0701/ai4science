import numpy as np

class LogisticRegressionNP:
    """Numpy implementation of Logistic Regression."""

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.w = np.zeros((self.input_dim, 1))
        self.b = np.zeros((1, 1))

    def _sigmoid(self, logit: np.ndarray) -> np.ndarray:
        """Sigmoid function with stabilization trick."""
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        return np.exp(logit_stable) / (np.exp(-logit_max) + np.exp(logit_stable))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Logistic regression forward pass (probabilities)."""
        logit = np.matmul(x, self.w) + self.b
        return self._sigmoid(logit)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_params(self) -> dict[str, np.ndarray]:
        """Return model parameters."""
        return {"w": self.w, "b": self.b}

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        """Set model parameters."""
        self.w = w
        self.b = b
