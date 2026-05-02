#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np


class LogisticRegression:
    """Numpy implementation of Logistic Regression."""

    def __init__(self, batch_size=64, lr=0.01, n_epochs=1000):
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def get_data(self, X_train, y_train, shuffle=True):
        """Get dataset and information."""
        self.X_train = X_train
        self.y_train = y_train

        # Get the numbers of examples and inputs.
        self.n_examples, self.n_inputs = self.X_train.shape

        if shuffle:
            idx = list(range(self.n_examples))
            random.shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]

    def _create_weights(self):
        """Create model weights and bias."""
        self.w = np.zeros(self.n_inputs).reshape(self.n_inputs, 1)
        self.b = np.zeros(1).reshape(1, 1)

    def _logit(self, X):
        """Logit: unnormalized log probability."""
        return np.matmul(X, self.w) + self.b

    def _sigmoid(self, logit):
        """Sigmoid function by stabilization trick.

        sigmoid(z) = 1 / (1 + exp(-z)) 
                   = exp(z) / (1 + exp(z)) * exp(z_max) / exp(z_max)
                   = exp(z - z_max) / (exp(-z_max) + exp(z - z_max)),
        where z is the logit, and z_max = z - max(0, z).
        """
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        return np.exp(logit_stable) / (np.exp(-logit_max) + np.exp(logit_stable))
    
    def _model(self, X):
        """Logistic regression model."""
        logit = self._logit(X)
        return self._sigmoid(logit)

    def _loss(self, y, logit):
        """Cross entropy loss by stabilizaiton trick.

        cross_entropy_loss(y, z) 
          = - 1/n * \sum_{i=1}^n y_i * log p(y_i = 1|x_i) + (1 - y_i) * log p(y_i = 0|x_i)
          = - 1/n * \sum_{i=1}^n y_i * (z_i - log(1 + exp(z_i))) + (1 - y_i) * (-log(1 + exp(z_i))),
        where z is the logit, z_max = z - max(0, z),
          log p(y = 1|x)
            = log (1 / (1 + exp(-z))) 
            = log (exp(z) / (1 + exp(z)))
            = z - log(1 + exp(z))
        and 
          log(1 + exp(z)) := logsumexp(z)
            = log(exp(0) + exp(z))
            = log(exp(0) + exp(z) * exp(z_max) / exp(z_max))
            = z_max + log(exp(-z_max) + exp(z - z_max)).
        """
        logit_max = np.maximum(0, logit)
        logit_stable = logit - logit_max
        logsumexp_stable = logit_max + np.log(np.exp(-logit_max) + np.exp(logit_stable))
        self.cross_entropy = -(y * (logit - logsumexp_stable) + (1 - y) * (-logsumexp_stable))
        return np.mean(self.cross_entropy)

    def _optimize(self, X, y):
        """Optimize by stochastic gradient descent."""
        m = X.shape[0]

        y_ = self._model(X) 
        dw = 1 / m * np.matmul(X.T, y_ - y)
        db = np.mean(y_ - y)

        for (param, grad) in zip([self.w, self.b], [dw, db]):
            param[:] = param - self.lr * grad

    def _fetch_batch(self):
        """Fetch batch dataset."""
        idx = list(range(self.n_examples))
        for i in range(0, self.n_examples, self.batch_size):
            idx_batch = idx[i:min(i + self.batch_size, self.n_examples)]
            yield (self.X_train.take(idx_batch, axis=0), 
                   self.y_train.take(idx_batch, axis=0))

    def fit(self):
        """Fit model."""
        self._create_weights()

        for epoch in range(self.n_epochs):
            total_loss = 0
            for X_train_b, y_train_b in self._fetch_batch():
                y_train_b = y_train_b.reshape((y_train_b.shape[0], -1))
                self._optimize(X_train_b, y_train_b)
                train_loss = self._loss(y_train_b, self._logit(X_train_b))
                total_loss += train_loss * X_train_b.shape[0]

            if epoch % 100 == 0:
                print('epoch {0}: training loss {1}'
                      .format(epoch, total_loss / self.n_examples))

        return self

    def get_coeff(self):
        """Get model coefficients."""
        return self.b, self.w.reshape((-1,))

    def predict(self, X):
        """Predict for new data."""
        return self._model(X).reshape((-1,))
