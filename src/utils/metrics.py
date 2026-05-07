"""
Evaluation utilities: per-task testing and aggregate "thesis" metrics.

* ``test_on_task``         -- evaluate the model on a single task test set.
* ``compute_thesis_metrics`` -- aggregate accuracy / PR-AUC matrices into
  the standard CL summary numbers (AvgAcc, ForgetAcc, AvgPR, ForgetPR,
  DiagPR).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score

# Suppress warnings that fire when a test window contains no fraud cases.
# This can happen in freeze/recurring datasets; affected metrics return NaN.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="No positive class found in y_true")


def test_on_task(model, X_test, y_test) -> dict:
    """
    Evaluate the model on a single task test set.

    Returns accuracy and PR-AUC; PR-AUC is NaN when fewer than 3 positives
    are present.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        y_true = y_test.squeeze(1).long().cpu().numpy()
        y_prob = probs.squeeze(1).cpu().numpy()

        accuracy = (preds == y_test).float().mean().item()

        n_pos = int((y_true == 1).sum())
        if n_pos <= 2:
            pr = float("nan")
        else:
            try:
                pr = average_precision_score(y_true, y_prob)
            except ValueError:
                pr = float("nan")

    return {"acc": accuracy, "pr_auc": pr}


def compute_thesis_metrics(acc_mat: np.ndarray, pr_mat: np.ndarray):
    """
    Compute standard thesis metrics from evaluation matrices.

    IMPORTANT ROBUSTNESS FIX:
    - In freeze/recurring drift, some tasks may have 0 (or very few) fraud
      cases in the test split.
    - In that case PR-AUC is undefined and may be NaN.
    - We therefore use NaN-safe aggregation for PR metrics.

    Returns:
        avg_acc, avg_forgetting_acc, avg_pr, avg_forgetting_pr, avg_diag_pr
    """
    n_tasks = acc_mat.shape[0]

    # Average performance after training on all tasks
    avg_acc = float(np.mean(acc_mat[n_tasks - 1]))
    avg_pr = float(np.nanmean(pr_mat[n_tasks - 1]))  # ignore NaNs

    # Forgetting / backward transfer-style metric on accuracy (always defined)
    forgetting_acc = 0.0
    for t in range(n_tasks - 1):
        forgetting_acc += acc_mat[n_tasks - 1, t] - acc_mat[t, t]
    avg_forgetting_acc = float(forgetting_acc / (n_tasks - 1))

    # Forgetting on PR-AUC (skip undefined comparisons)
    pr_diffs = []
    for t in range(n_tasks - 1):
        final_val = pr_mat[n_tasks - 1, t]
        diag_val = pr_mat[t, t]
        if not (np.isnan(final_val) or np.isnan(diag_val)):
            pr_diffs.append(final_val - diag_val)

    avg_forgetting_pr = (
        float(np.mean(pr_diffs)) if len(pr_diffs) > 0 else float("nan")
    )

    # Plasticity: average peak performance per task (diagonal = right after
    # training on that task).
    avg_diag_pr = float(np.nanmean(np.diag(pr_mat)))

    return avg_acc, avg_forgetting_acc, avg_pr, avg_forgetting_pr, avg_diag_pr
