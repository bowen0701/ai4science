import numpy as np
from dl_eng.models.logistic_regression_np import LogisticRegressionNP
from dl_eng.learners.numpy import NumPyLearner
from dl_eng.infra.logger import setup_logger

logger = setup_logger("LogisticRegressionTraining")

def train_numpy():
    logger.info("Starting NumPy Logistic Regression Training")
    # Generate dummy data for binary classification
    X = np.random.randn(200, 2)
    # Decision boundary: x1 + x2 > 0
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

    model = LogisticRegressionNP(input_dim=2)
    
    # Custom Cross Entropy loss for Logistic Regression
    def cross_entropy_loss(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    learner = NumPyLearner(model, lr=0.5, loss_fn=cross_entropy_loss)

    for epoch in range(100):
        metrics = learner.train_step((X, y))
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")

    logger.info("NumPy Logistic Training Complete")
    logger.info(f"Learned params: {model.get_params()}")
    
    # Test accuracy
    preds = (model.forward(X) > 0.5).astype(float)
    accuracy = np.mean(preds == y)
    logger.info(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_numpy()
