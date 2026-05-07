"""
PackNet -- parameter isolation via iterative pruning.

Three helper functions implement PackNet's parameter isolation:

    1) ``init_frozen_mask``              -- initialise mask with all weights free.
    2) ``apply_pruning_and_update_mask`` -- prune + freeze after each task.
    3) ``report_mask_capacity``          -- print remaining free capacity.

After each task, the lowest-magnitude free weights are pruned (zeroed) and
the remaining free weights are frozen. Each task owns a private,
non-overlapping subset of parameters, preventing cross-task interference.

PackNet operates in the Task-IL setting; other methods use Domain-IL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


def init_frozen_mask(model):
    """All weights start as FREE (False = not frozen)."""
    return {
        name: torch.zeros_like(param.data, dtype=torch.bool)
        for name, param in model.named_parameters()
    }


def apply_pruning_and_update_mask(model, frozen_mask, prune_ratio: float = 0.1):
    """
    Post-task step: zero the lowest-magnitude free weights, then freeze the rest.

    Args:
        model:       Trained model after the current task.
        frozen_mask: Current mask (updated in place).
        prune_ratio: Fraction of free weights to prune.

    Returns:
        Updated frozen_mask.
    """
    for name, param in model.named_parameters():
        w = param.data
        free_mask = ~frozen_mask[name]

        if free_mask.sum().item() == 0:
            continue

        free_weights_abs = w[free_mask].abs()
        n_free = free_weights_abs.numel()
        n_prune = int(prune_ratio * n_free)

        if n_prune <= 0:
            frozen_mask[name][free_mask] = True
            continue

        threshold = torch.kthvalue(free_weights_abs, k=min(n_prune, n_free)).values.item()
        w[free_mask & (w.abs() <= threshold)] = 0.0
        frozen_mask[name][free_mask & (w != 0.0)] = True

    return frozen_mask


def report_mask_capacity(model, frozen_mask) -> None:
    """Print a capacity report: frozen vs free parameters."""
    total = frozen = free_nz = free_z = 0
    for name, param in model.named_parameters():
        w, fm = param.data, frozen_mask[name]
        free_mask = ~fm
        total += w.numel()
        frozen += fm.sum().item()
        free_nz += (free_mask & (w != 0.0)).sum().item()
        free_z += (free_mask & (w == 0.0)).sum().item()
    free = total - frozen
    print(
        f"  [Capacity] Total={total} | Frozen={frozen} ({frozen/total:.0%}) | "
        f"Free={free} ({free/total:.0%}) [non-zero={free_nz}, zero={free_z}]"
    )


def train_with_packnet(
    model,
    X_train,
    y_train,
    frozen_mask,
    epochs: int = 20,
    lr: float = 0.01,
):
    """
    Identical to ``train_on_task`` with one addition: gradients for frozen
    parameters are zeroed before the optimizer step so only free weights
    are updated.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()

        # Zero out gradients for frozen weights so the optimizer cannot update them.
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad[frozen_mask[name]] = 0.0

        optimizer.step()

    model.eval()
    return model
