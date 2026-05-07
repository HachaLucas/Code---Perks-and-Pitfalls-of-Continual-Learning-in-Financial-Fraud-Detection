"""
Disk persistence for per-(dataset, method) accuracy and PR-AUC matrices.

These helpers were originally inlined in the runner cell as
``_ckpt_paths`` / ``_save_ckpt`` / ``_load_ckpt``. They live here so that
both ``scripts/run_experiments.py`` and the demo notebook can reuse them
to load saved results for visualisation.

Reasoning for this location: checkpoints are a generic I/O concern (they
do not depend on any specific method or dataset), so ``src/utils`` is the
natural home. Keeping them here also makes it easy for the visualisation
code to read results back without having to import the runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def ckpt_paths(results_dir: Path | str, ds_name: str, method: str) -> Tuple[Path, Path]:
    """Return (acc_path, pr_path) for one (dataset, method) pair."""
    results_dir = Path(results_dir)
    d = results_dir / ds_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{method}_acc.npy", d / f"{method}_pr.npy"


def load_ckpt(results_dir: Path | str, ds_name: str, method: str):
    """Return (acc_mat, pr_mat) if both files exist, else None."""
    acc_p, pr_p = ckpt_paths(results_dir, ds_name, method)
    if acc_p.exists() and pr_p.exists():
        return np.load(acc_p), np.load(pr_p)
    return None


def save_ckpt(
    results_dir: Path | str,
    ds_name: str,
    method: str,
    acc_m: np.ndarray,
    pr_m: np.ndarray,
) -> None:
    """Save (acc_mat, pr_mat) under ``results_dir/<ds_name>/<method>_*.npy``."""
    acc_p, pr_p = ckpt_paths(results_dir, ds_name, method)
    np.save(acc_p, acc_m)
    np.save(pr_p, pr_m)
