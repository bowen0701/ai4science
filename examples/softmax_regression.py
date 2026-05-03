import torch
import torch.nn as nn
import torch.optim as optim
from dl_eng.models.softmax_regression import SoftmaxRegression
from dl_eng.learners.torch import TorchLearner
from dl_eng.infra.logger import setup_logger

logger = setup_logger("SoftmaxRegressionTraining")

def train_neural():
    logger.info("Starting PyTorch Softmax Regression Training")
    # Generate dummy data for multi-class classification (3 classes)
    X = torch.randn(300, 4)
    y = torch.randint(0, 3, (300,))

    model = SoftmaxRegression(input_dim=4, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    learner = TorchLearner(model, optimizer, loss_fn)

    for epoch in range(100):
        metrics = learner.train_step((X, y))
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")

    logger.info("PyTorch Softmax Training Complete")

if __name__ == "__main__":
    train_neural()
