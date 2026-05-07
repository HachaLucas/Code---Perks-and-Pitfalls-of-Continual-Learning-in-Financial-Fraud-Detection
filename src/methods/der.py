"""
Dark Experience Replay (DER++) training function.

DER is a replay-based method that stores not just (X, y) but also the
LOGITS the model produced at the time of storage -- the "dark knowledge".

WHY LOGITS?
Hard labels (0/1) discard information. A logit of 3.8 vs 0.2 both round
to "fraud", but they express very different levels of model confidence.
Storing logits gives a richer supervision signal for future tasks.

DER++ LOSS (three terms):

    L = task_loss
      + alpha * MSE(current logits on buf_X, stored logits)
      + beta  * BCE(current logits on buf_X, stored labels)

    - task_loss:   learn the current task (BCEWithLogitsLoss on new data)
    - alpha term:  distillation -- stay close to what the model once believed
                   (dark knowledge; richer than hard labels)
    - beta term:   hard-label anchor -- do not drift from the original ground
                   truth (DER++ adds this on top of plain DER to stabilise)

BUFFER MANAGEMENT (handled by the caller, not here):
    buf_X      -- raw feature tensors from past tasks
    buf_logits -- logits the model produced when those samples were first seen
    buf_y      -- original ground-truth labels

On Task 0, ``buf_X`` is empty so both replay terms are skipped automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def train_with_der(
    model,
    X_train,
    y_train,
    buf_X,
    buf_logits,
    buf_y,
    alpha: float = 0.5,
    beta: float = 0.5,
    epochs: int = 20,
    lr: float = 0.01,
):
    """
    Args:
        model:              Neural network to train.
        X_train, y_train:   Current task data.
        buf_X:              Buffered feature tensors from past tasks.
        buf_logits:         Logits the model produced when buf_X was first seen.
        buf_y:              Ground-truth labels for buf_X.
        alpha:              Weight on the distillation (dark-knowledge) loss.
        beta:               Weight on the hard-label replay loss.
        epochs:             Training epochs.
        lr:                 Learning rate.

    Returns:
        model:      Updated model.
        new_logits: Logits produced on X_train at the END of training.
                    Store these in the buffer alongside X_train and y_train.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    has_buffer = buf_X.numel() > 0

    for _ in range(epochs):
        optimizer.zero_grad()

        # Current task loss
        logits_new = model(X_train)
        task_loss = criterion(logits_new, y_train)

        loss = task_loss

        # Replay terms (skipped on Task 0 when buffer is empty)
        if has_buffer:
            logits_buf = model(buf_X)

            # Alpha term: distillation -- reproduce stored dark knowledge.
            distill_loss = mse(logits_buf, buf_logits)

            # Beta term: hard-label anchor -- match original ground truth.
            hard_loss = criterion(logits_buf, buf_y)

            loss = loss + alpha * distill_loss + beta * hard_loss

        loss.backward()
        optimizer.step()

    model.eval()

    # Compute and return logits on the current task data AFTER training.
    # These will be stored in the buffer for future tasks.
    with torch.no_grad():
        new_logits = model(X_train)

    return model, new_logits
