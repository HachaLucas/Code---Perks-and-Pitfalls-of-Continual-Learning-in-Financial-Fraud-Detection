"""
Experience replay.

Trains on a mix of current-task data and a fixed-size buffer of past
transactions. On the first task the buffer is empty and training proceeds
on new data only. Buffer management (sampling and trimming to
``MEMORY_SIZE``) is handled by the caller (the runner).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def train_with_replay(
    model,
    X_new,
    y_new,
    buffer_X,
    buffer_y,
    epochs: int = 20,
    lr: float = 0.01,
):
    """
    Args:
        model:              Neural network to train.
        X_new, y_new:       Current task data.
        buffer_X, buffer_y: Stored samples from previous tasks
                            (empty on task 0).
        epochs:             Training epochs.
        lr:                 Learning rate for Adam.

    Returns:
        model: Updated model trained on current + replay data.
    """
    if buffer_X.numel() > 0:
        X_combined = torch.cat((X_new, buffer_X), dim=0)
        y_combined = torch.cat((y_new, buffer_y), dim=0)
    else:
        X_combined = X_new
        y_combined = y_new

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_combined)
        loss = criterion(logits, y_combined)
        loss.backward()
        optimizer.step()

    model.eval()
    return model
