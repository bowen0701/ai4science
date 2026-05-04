#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class LogisticRegression(nn.Module):
    """PyTorch implementation of Linear Regression."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Logistic regression.
        self.fc1 = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
