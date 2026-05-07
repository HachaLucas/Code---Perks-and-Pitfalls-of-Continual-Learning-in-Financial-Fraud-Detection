"""
Dataset loading for the continual-learning fraud-detection experiments.

This module wraps the data-loading logic from the original Jupyter notebook
in a small ``DataLoader`` class. The class owns its own dataset path,
StandardScaler, and per-task tensor cache, so multiple datasets can be
processed in sequence without leaking state between them (which is what
the global ``_df_global`` / ``_scaler`` / ``_data_cache`` variables in the
notebook were doing implicitly).

Behaviour is identical to the notebook:

* Periods in the CSV are 1-based; ``task_id`` is 0-based.
* Stratified train/test split is used when both classes have >= 2 samples,
  and falls back to a non-stratified split otherwise.
* SMOTE is applied to the training fold only when the minority class has
  >= 6 samples (k_neighbors=5 + 1).
* The scaler is fit on the **first** training chunk and reused for every
  subsequent task in the same dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Feature columns expected in every CSV. Kept module-level so callers can
# import it for inspection / sanity checks without needing a DataLoader.
FEATURE_COLS = [
    "time_hour", "amount", "is_remote",
    "velocity_1h", "velocity_24h", "amount_zscore",
    "is_foreign", "hours_since_last",
]


class DataLoader:
    """Encapsulates all per-dataset state (path, scaler, cache).

    Parameters
    ----------
    data_path : str | Path
        Path to a CSV file with the columns ``FEATURE_COLS`` plus
        ``period`` and ``is_fraud``.
    feature_cols : list[str], optional
        Override the default feature list.
    test_size : float, default 0.2
        Fraction held out for the test split.
    random_state : int, default 42
        Seed for the train/test split and SMOTE.
    """

    def __init__(
        self,
        data_path: str | Path,
        feature_cols: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.data_path = str(data_path)
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
        self.test_size = test_size
        self.random_state = random_state

        self._df: pd.DataFrame | None = None
        self._scaler: StandardScaler | None = None
        self._cache: dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    # -- public API -----------------------------------------------------

    def load(self) -> None:
        """Read the CSV from disk (idempotent)."""
        if self._df is not None:
            return
        print("Loading dataset...")
        self._df = pd.read_csv(self.data_path)
        print(
            f"Loaded {len(self._df):,} rows | fraud rate: "
            f"{self._df['is_fraud'].mean()*100:.3f}%"
        )

    def get_data_for_task(
        self,
        task_id: int,
        split: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the (X, y) tensors for one task / period.

        IMPORTANT ROBUSTNESS FIX:
        - Some drift datasets (freeze/recurring) can contain periods with
          0 or 1 fraud sample.
        - Stratified splitting requires at least 2 samples in EACH class.
        - We stratify only when valid; otherwise we fall back to a normal
          split.
        """
        if (task_id, split) in self._cache:
            return self._cache[(task_id, split)]

        self.load()
        assert self._df is not None  # for type checkers

        period = task_id + 1  # periods in CSV are 1-based
        period_df = self._df[self._df["period"] == period].copy()

        # Safety check: empty period -> fail fast with a clear message.
        if len(period_df) == 0:
            raise ValueError(
                f"Period/task {period} is empty in the dataset. "
                f"Check your period construction."
            )

        X_all = period_df[self.feature_cols].values.astype("float32")
        y_all = period_df["is_fraud"].values.astype("float32")

        # --- Safe stratify logic ---
        n_pos = int((y_all == 1).sum())
        n_neg = int((y_all == 0).sum())
        use_stratify = (n_pos >= 2) and (n_neg >= 2)

        if not use_stratify:
            print(
                f"   [Split Warning] Task {task_id}: stratify disabled "
                f"(n_pos={n_pos} < 2 required)"
            )

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all if use_stratify else None,
        )

        # --- SMOTE oversampling (training data only) ---
        # Applied after train/test split and before scaling to prevent leakage.
        n_pos_train = int((y_train == 1).sum())
        if n_pos_train < 6:
            print(
                f"   [SMOTE Warning] Task {task_id}: skipping SMOTE "
                f"(n_pos={n_pos_train} < 6 required)"
            )
        else:
            smote = SMOTE(
                random_state=self.random_state,
                k_neighbors=5,
                sampling_strategy="auto",
            )
            X_train_raw, y_train = smote.fit_resample(X_train_raw, y_train)

        # Fit scaler once on the first training chunk, reuse for all later tasks.
        if self._scaler is None:
            self._scaler = StandardScaler().fit(X_train_raw)

        X_train = self._scaler.transform(X_train_raw)
        X_test = self._scaler.transform(X_test_raw)

        self._cache[(task_id, "train")] = (
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        self._cache[(task_id, "test")] = (
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).unsqueeze(1),
        )
        return self._cache[(task_id, split)]

    def reset(self) -> None:
        """Drop the in-memory dataframe, scaler and cache."""
        self._df = None
        self._scaler = None
        self._cache = {}
