import numpy as np

class LinearRegressionNP:
    """Numpy implementation of Linear Regression."""

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.w = np.zeros((self.input_dim, 1))
        self.b = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Linear regression forward pass."""
        return np.matmul(x, self.w) + self.b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_params(self) -> dict[str, np.ndarray]:
        """Return model parameters."""
        return {"w": self.w, "b": self.b}

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        """Set model parameters."""
        self.w = w
        self.b = b
