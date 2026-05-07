"""
Synaptic Intelligence (SI).

Regularization-based continual-learning method. Importance is accumulated
ONLINE during training (rather than computed post-hoc as in EWC).

During training:
    small_omega_i += - grad_task_i * delta_theta_i

After each task:
    Omega_i += max(0, small_omega_i / (delta_theta_i_total^2 + xi))

During future tasks:
    L_total = L_task + c * sum_i Omega_i * (theta_i - theta_star_i)^2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class SynapticIntelligence:
    def __init__(self, xi: float = 1e-3) -> None:
        self.omega: dict = {}        # cumulative importance Omega
        self.theta_star: dict = {}   # parameter snapshot after last completed task
        self.xi = xi                 # damping term

    def register_task(self, model, small_omega, prev_params) -> None:
        """
        Update cumulative importance Omega and store parameter snapshot
        after completing one task.

        Args:
            model:        Model after training on the current task.
            small_omega:  Path integral accumulated during the current task.
            prev_params:  Parameter snapshot from the start of the current task.
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            current_param = param.detach()
            delta = current_param - prev_params[name]

            # SI importance for this task
            task_importance = small_omega[name] / (delta.pow(2) + self.xi)

            # Only positive importance should consolidate weights.
            # Negative values are usually noise or parameters that increased the loss.
            task_importance = torch.clamp(task_importance, min=0.0)

            if name not in self.omega:
                self.omega[name] = torch.zeros_like(current_param)

            self.omega[name] = self.omega[name] + task_importance.detach()
            self.theta_star[name] = current_param.clone()

    def penalty(self, model):
        """SI regularisation penalty. Returns zero before any task is registered."""
        loss = next(model.parameters()).new_tensor(0.0)

        for name, param in model.named_parameters():
            if name in self.omega:
                omega = self.omega[name].to(param.device)
                theta_star = self.theta_star[name].to(param.device)
                loss = loss + (omega * (param - theta_star).pow(2)).sum()

        return loss


def train_with_si(
    model,
    X_train,
    y_train,
    si: SynapticIntelligence,
    importance_factor: float = 500,
    epochs: int = 20,
    lr: float = 0.01,
):
    """
    Train on the current task with SI regularisation.

    Important detail:
    - The model update uses ``total_loss = task_loss + c * SI_penalty``.
    - The SI path integral ``small_omega`` uses the gradient of ``task_loss`` only.

    Returns:
        model:        Updated model.
        small_omega:  Per-weight path-integral accumulator for this task.
        prev_params:  Parameter snapshot from the start of this task.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Snapshot at start of task
    prev_params = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    # Path integral accumulator for current task
    small_omega = {
        n: torch.zeros_like(p.detach())
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    for _ in range(epochs):
        optimizer.zero_grad()

        # Parameters before optimizer step
        params_before = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        logits = model(X_train)
        task_loss = criterion(logits, y_train)
        si_loss = si.penalty(model)
        total_loss = task_loss + importance_factor * si_loss

        # 1. Gradients of task_loss only (for SI omega).
        #    retain_graph=True because we still need backward() on total_loss.
        task_grads = torch.autograd.grad(
            task_loss,
            trainable_params,
            retain_graph=True,
            allow_unused=True,
        )

        # 2. total_loss for the actual parameter update.
        total_loss.backward()
        optimizer.step()

        # 3. Accumulate SI path integral using task gradients only.
        with torch.no_grad():
            for name, param, task_grad in zip(trainable_names, trainable_params, task_grads):
                if task_grad is not None:
                    delta = param.detach() - params_before[name]
                    small_omega[name] += -task_grad.detach() * delta

    model.eval()
    return model, small_omega, prev_params
