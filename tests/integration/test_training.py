import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from dl_eng.models.linear_regression_np import LinearRegressionNP
from dl_eng.models.logistic_regression_np import LogisticRegressionNP
from dl_eng.models.softmax_regression import SoftmaxRegression
from dl_eng.learners.numpy import NumPyLearner
from dl_eng.learners.torch import TorchLearner

def test_linear_regression_numpy():
    X = np.random.randn(100, 5)
    y = np.dot(X, np.array([1, 2, 3, 4, 5])) + 0.5
    y = y.reshape(-1, 1)

    model = LinearRegressionNP(input_dim=5)
    learner = NumPyLearner(model, lr=0.1)

    for epoch in range(10):
        metrics = learner.train_step((X, y))
    
    assert "loss" in metrics
    assert metrics["loss"] < 50  # Basic sanity check

def test_logistic_regression_numpy():
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

    model = LogisticRegressionNP(input_dim=2)
    
    def cross_entropy_loss(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    learner = NumPyLearner(model, lr=0.5, loss_fn=cross_entropy_loss)

    for epoch in range(10):
        metrics = learner.train_step((X, y))
    
    assert "loss" in metrics

def test_softmax_regression_torch():
    X = torch.randn(300, 4)
    y = torch.randint(0, 3, (300,))

    model = SoftmaxRegression(input_dim=4, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    learner = TorchLearner(model, optimizer, loss_fn)

    for epoch in range(10):
        metrics = learner.train_step((X, y))
    
    assert "loss" in metrics
