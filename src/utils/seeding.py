"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for ``random``, ``numpy`` and ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
