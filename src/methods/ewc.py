"""
Elastic Weight Consolidation (EWC).

Classical regularization-based continual-learning method. After each task,
EWC estimates the importance of each parameter using the diagonal Fisher
Information Matrix and stores both:

    1. a snapshot of the parameters after the task, theta*_t
    2. the task-specific Fisher estimate, F_t

During future tasks, the loss is:

    L = L_task + lambda * sum_{k<t} sum_i F_{k,i} * (theta_i - theta*_{k,i})^2

Thus, each previous task contributes its own quadratic penalty term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class EWC:
    """Classical EWC: stores a separate Fisher matrix and parameter snapshot per task."""

    def __init__(self) -> None:
        self.task_params: list[dict[str, torch.Tensor]] = []
        self.task_fishers: list[dict[str, torch.Tensor]] = []

    def update(self, model, dataset, fisher_samples: int = 1000) -> None:
        """
        Compute the diagonal Fisher for the just-trained task and store it
        together with a snapshot of the current parameters.
        """
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        X_data, y_data = dataset

        n_samples = min(fisher_samples, len(X_data))
        idx = torch.randperm(len(X_data), device=X_data.device)[:n_samples]
        X_sub, y_sub = X_data[idx], y_data[idx]

        fisher = {
            name: torch.zeros_like(param.detach())
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for i in range(n_samples):
            model.zero_grad()

            logits = model(X_sub[i].unsqueeze(0))
            loss = criterion(logits, y_sub[i].unsqueeze(0))
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)

        for name in fisher:
            fisher[name] /= n_samples

        params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self.task_fishers.append(fisher)
        self.task_params.append(params)

    def penalty(self, model):
        """
        Classical EWC penalty:
        sum over previous tasks k and parameters i:
            F_{k,i} * (theta_i - theta*_{k,i})^2
        """
        loss = next(model.parameters()).new_tensor(0.0)

        if len(self.task_params) == 0:
            return loss

        for params, fisher in zip(self.task_params, self.task_fishers):
            for name, param in model.named_parameters():
                if name in fisher:
                    f = fisher[name].to(param.device)
                    theta_star = params[name].to(param.device)
                    loss = loss + (f * (param - theta_star).pow(2)).sum()

        return loss


def train_with_ewc(
    model,
    X_train,
    y_train,
    ewc: EWC,
    importance_factor: float = 500,
    epochs: int = 20,
    lr: float = 0.01,
):
    """
    Args:
        model:             Neural network to train.
        X_train, y_train:  Current task data.
        ewc:               EWC object storing Fisher matrices and parameter snapshots for past tasks.
        importance_factor: Lambda -- penalty strength.
        epochs:            Training epochs.
        lr:                Learning rate.

    Returns:
        model: Updated model.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        task_loss = criterion(logits, y_train)

        ewc_loss = ewc.penalty(model)

        loss = task_loss + (importance_factor * ewc_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    return model