"""
Model architecture used across all continual-learning methods.

A fixed 3-layer MLP. The architecture is kept constant so that performance
differences reflect the CL method, not the model.

* BatchNorm stabilizes training under covariate drift.
* Dropout reduces overfitting under class imbalance.
* The output is a raw logit (no Sigmoid); ``BCEWithLogitsLoss`` handles
  the sigmoid internally during training.
"""

from __future__ import annotations

import torch.nn as nn


class FraudDetector(nn.Module):
    def __init__(self, input_features: int = 8, dropout: float = 0.3) -> None:
        super().__init__()

        self.layer1 = nn.Linear(input_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)

        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout)

        # Single shared output head: Domain-IL setup, label space never
        # changes and no task ID is passed at inference time.
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu(self.bn2(self.layer2(x))))
        x = self.output_layer(x)
        return x
