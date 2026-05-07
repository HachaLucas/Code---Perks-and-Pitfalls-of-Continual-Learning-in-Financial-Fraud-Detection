"""
Naive current-task training baseline.

For each task, a freshly initialized model is trained only on that task's
training data using Adam + BCEWithLogitsLoss. No parameters, replay data,
optimizer state, or regularization information are transferred between tasks.

BCEWithLogitsLoss applies Sigmoid internally for numerical stability.
"""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


def train_on_task(model, X_train, y_train, epochs: int = 20, lr: float = 0.01):
    """
    Args:
        model:   Neural network to train.
        X_train: torch.Tensor (N, n_features) -- transaction features.
        y_train: torch.Tensor (N, 1)          -- fraud labels (0 or 1).
        epochs:  Training epochs.
        lr:      Learning rate for Adam.

    Returns:
        model: The trained model (updated in place).
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    return model
