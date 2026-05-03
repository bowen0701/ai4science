import numpy as np
import torch
from dl_eng.models.linear_regression_np import LinearRegressionNP
from dl_eng.models.linear_regression import LinearRegression
from dl_eng.learners.numpy import NumPyLearner
from dl_eng.infra.logger import setup_logger

logger = setup_logger("LinearRegressionTraining")

def train_numpy():
    logger.info("Starting NumPy Linear Regression Training")
    # Generate dummy data
    X = np.random.randn(100, 5)
    y = np.dot(X, np.array([1, 2, 3, 4, 5])) + 0.5
    y = y.reshape(-1, 1)

    model = LinearRegressionNP(input_dim=5)
    learner = NumPyLearner(model, lr=0.1)

    for epoch in range(100):
        metrics = learner.train_step((X, y))
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")

    logger.info("NumPy Training Complete")
    logger.info(f"Learned params: {model.get_params()}")

if __name__ == "__main__":
    train_numpy()
