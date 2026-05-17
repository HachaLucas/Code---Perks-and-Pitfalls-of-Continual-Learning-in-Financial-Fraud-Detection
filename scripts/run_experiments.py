"""
CLI entry point for the multi-dataset continual-learning runner.

Usage
-----
From the project root::

    python -m scripts.run_experiments --config config/experiment.yaml

Or, if you prefer running it directly::

    python scripts/run_experiments.py --config config/experiment.yaml

The runner loads all hyperparameters from the YAML config, then loops
over the configured datasets and methods. Per-(dataset, method) accuracy
and PR-AUC matrices are checkpointed to ``<results_dir>/<ds_name>/`` so
re-running only re-trains the missing combinations.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Make ``src`` importable when the script is run directly. Putting the
# project root on sys.path keeps imports identical whether the user
# invokes this as ``python scripts/run_experiments.py`` or
# ``python -m scripts.run_experiments``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DataLoader  # noqa: E402
from src.methods import (  # noqa: E402
    EWC,
    FraudDetector,
    SynapticIntelligence,
    apply_pruning_and_update_mask,
    init_frozen_mask,
    train_on_task,
    train_with_der,
    train_with_ewc,
    train_with_packnet,
    train_with_replay,
    train_with_si,
)
from src.utils import (  # noqa: E402
    compute_thesis_metrics,
    load_ckpt,
    save_ckpt,
    set_seed,
    test_on_task,
)


# --------------------------------------------------------------------- #
# Dataset registry                                                      #
# --------------------------------------------------------------------- #

def build_datasets(base_dir: Path) -> dict:
    """Build the dataset registry, parameterised by the dataset folder."""
    base_dir = Path(base_dir)
    return {
        # AXIS 0: BASELINE (1 dataset)
        "baseline": {
            "path": base_dir / "baseline_pattern_A_stationary.csv",
            "axis": 0, "label": "Baseline",
        },

        # AXIS 1: DRIFT MAGNITUDE SUDDEN — 5 datasets
        "axis1_delta0.3_sudden": {"path": base_dir / "axis1_delta0.3_sudden.csv", "axis": 1, "label": "Sudden δ=0.3", "scale": 0.3},
        "axis1_delta0.6_sudden": {"path": base_dir / "axis1_delta0.6_sudden.csv", "axis": 1, "label": "Sudden δ=0.6", "scale": 0.6},
        "axis1_delta0.9_sudden": {"path": base_dir / "axis1_delta0.9_sudden.csv", "axis": 1, "label": "Sudden δ=0.9", "scale": 0.9},
        "axis1_delta1.2_sudden": {"path": base_dir / "axis1_delta1.2_sudden.csv", "axis": 1, "label": "Sudden δ=1.2", "scale": 1.2},
        "axis1_delta1.5_sudden": {"path": base_dir / "axis1_delta1.5_sudden.csv", "axis": 1, "label": "Sudden δ=1.5", "scale": 1.5},

        # AXIS 2: DRIFT MAGNITUDE INCREDMENTAL — 5 datasets
        "axis2_delta0.3_incremental": {"path": base_dir / "axis2_delta0.3_incremental.csv", "axis": 2, "label": "Incremental δ=0.3", "scale": 0.3},
        "axis2_delta0.6_incremental": {"path": base_dir / "axis2_delta0.6_incremental.csv", "axis": 2, "label": "Incremental δ=0.6", "scale": 0.6},
        "axis2_delta0.9_incremental": {"path": base_dir / "axis2_delta0.9_incremental.csv", "axis": 2, "label": "Incremental δ=0.9", "scale": 0.9},
        "axis2_delta1.2_incremental": {"path": base_dir / "axis2_delta1.2_incremental.csv", "axis": 2, "label": "Incremental δ=1.2", "scale": 1.2},
        "axis2_delta1.5_incremental": {"path": base_dir / "axis2_delta1.5_incremental.csv", "axis": 2, "label": "Incremental δ=1.5", "scale": 1.5},

        # AXIS 3: FREEZE DURATION — 6 datasets
        "axis3_freeze_k1": {"path": base_dir / "axis3_freeze_k1.csv", "axis": 3, "label": "Freeze k=1", "k": 1},
        "axis3_freeze_k2": {"path": base_dir / "axis3_freeze_k2.csv", "axis": 3, "label": "Freeze k=2", "k": 2},
        "axis3_freeze_k3": {"path": base_dir / "axis3_freeze_k3.csv", "axis": 3, "label": "Freeze k=3", "k": 3},
        "axis3_freeze_k4": {"path": base_dir / "axis3_freeze_k4.csv", "axis": 3, "label": "Freeze k=4", "k": 4},
        "axis3_freeze_k5": {"path": base_dir / "axis3_freeze_k5.csv", "axis": 3, "label": "Freeze k=5", "k": 5},
        "axis3_freeze_k6": {"path": base_dir / "axis3_freeze_k6.csv", "axis": 3, "label": "Freeze k=6", "k": 6},

        # AXIS 4: OFFSET — 6 datasets
        "axis4_offset0.2":  {"path": base_dir / "axis4_offset0.2.csv",  "axis": 4, "label": "Offset 0.2",  "offset": 0.2},
        "axis4_offset1.0":  {"path": base_dir / "axis4_offset1.0.csv",  "axis": 4, "label": "Offset 1.0",  "offset": 1.0},
        "axis4_offset2.0":  {"path": base_dir / "axis4_offset2.0.csv",  "axis": 4, "label": "Offset 2.0",  "offset": 2.0},
        "axis4_offset4.0":  {"path": base_dir / "axis4_offset4.0.csv",  "axis": 4, "label": "Offset 4.0",  "offset": 4.0},
        "axis4_offset8.0":  {"path": base_dir / "axis4_offset8.0.csv",  "axis": 4, "label": "Offset 8.0",  "offset": 8.0},
        "axis4_offset40.0": {"path": base_dir / "axis4_offset40.0.csv", "axis": 4, "label": "Offset 40.0", "offset": 40.0},

        # AXIS 5: PATTERN ROTATION — 1 dataset
        "axis5_pattern_rotation": {"path": base_dir / "axis5_pattern_rotation.csv", "axis": 5, "label": "Pattern Rotation", "scale": 1.0},
    }


# --------------------------------------------------------------------- #
# Per-method runners                                                    #
# --------------------------------------------------------------------- #

def _run_naive(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    for task_id in range(n_tasks):
        model = FraudDetector(input_features=cfg["n_features"])

        X_train, y_train = loader.get_data_for_task(task_id, "train")
        model = train_on_task(
            model,
            X_train,
            y_train,
            epochs=cfg["n_epochs"],
            lr=cfg["learning_rate"],
        )

        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]

    return acc, pr


def _run_replay(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    model = FraudDetector(input_features=cfg["n_features"])
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    buf_X = torch.FloatTensor([])
    buf_y = torch.FloatTensor([])

    for task_id in range(n_tasks):
        X_train, y_train = loader.get_data_for_task(task_id, "train")
        model = train_with_replay(model, X_train, y_train, buf_X, buf_y,
                                  epochs=cfg["n_epochs"], lr=cfg["learning_rate"])

        idx = torch.randperm(len(X_train))[:cfg["memory_size"]]
        new_X = X_train[idx]
        new_y = y_train[idx]
        if buf_X.numel() == 0:
            buf_X, buf_y = new_X.clone(), new_y.clone()
        else:
            buf_X = torch.cat((buf_X, new_X), dim=0)
            buf_y = torch.cat((buf_y, new_y), dim=0)
        if buf_X.size(0) > cfg["memory_size"]:
            keep = torch.randperm(len(buf_X))[:cfg["memory_size"]]
            buf_X, buf_y = buf_X[keep], buf_y[keep]

        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]
    return acc, pr


def _run_full_replay(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    model = FraudDetector(input_features=cfg["n_features"])
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    fr_buf_X = torch.FloatTensor([])
    fr_buf_y = torch.FloatTensor([])

    for task_id in range(n_tasks):
        X_train, y_train = loader.get_data_for_task(task_id, "train")
        model = train_with_replay(model, X_train, y_train, fr_buf_X, fr_buf_y,
                                  epochs=cfg["n_epochs"], lr=cfg["learning_rate"])

        if fr_buf_X.numel() == 0:
            fr_buf_X, fr_buf_y = X_train.clone(), y_train.clone()
        else:
            fr_buf_X = torch.cat((fr_buf_X, X_train), dim=0)
            fr_buf_y = torch.cat((fr_buf_y, y_train), dim=0)

        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]
    return acc, pr


def _run_ewc(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    model = FraudDetector(input_features=cfg["n_features"])
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    ewc = EWC()  # stores task-specific Fisher matrices and parameter snapshots

    for task_id in range(n_tasks):
        X_train, y_train = loader.get_data_for_task(task_id, "train")
        model = train_with_ewc(
            model, X_train, y_train, ewc,
            importance_factor=cfg["ewc_lambda"],
            epochs=cfg["n_epochs"], lr=cfg["learning_rate"],
        )
        ewc.update(model, (X_train, y_train), fisher_samples=cfg["fisher_samples"])
        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]
    return acc, pr


def _run_packnet(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    model = FraudDetector(input_features=cfg["n_features"])
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    frozen_mask = init_frozen_mask(model)

    for task_id in range(n_tasks):
        X_train, y_train = loader.get_data_for_task(task_id, "train")

        model = train_with_packnet(
            model, X_train, y_train, frozen_mask,
            epochs=cfg["n_epochs"],
            lr=cfg["learning_rate"]
        )
        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]
        frozen_mask = apply_pruning_and_update_mask(
            model, frozen_mask, prune_ratio=cfg["packnet_prune_ratio"]
        )

    return acc, pr


def _run_si(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    model = FraudDetector(input_features=cfg["n_features"])
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    si = SynapticIntelligence(xi=cfg["si_xi"])

    for task_id in range(n_tasks):
        X_train, y_train = loader.get_data_for_task(task_id, "train")
        model, small_omega, prev_params = train_with_si(
            model, X_train, y_train, si,
            importance_factor=cfg["si_lambda"],
            epochs=cfg["n_epochs"], lr=cfg["learning_rate"],
        )
        si.register_task(model, small_omega, prev_params)
        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]
    return acc, pr


def _run_der(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    n_tasks = cfg["n_tasks"]
    mem_size = cfg["memory_size"]
    model = FraudDetector(input_features=cfg["n_features"])
    acc = np.zeros((n_tasks, n_tasks))
    pr = np.zeros((n_tasks, n_tasks))
    der_buf_X = torch.FloatTensor([])
    der_buf_logits = torch.FloatTensor([])
    der_buf_y = torch.FloatTensor([])

    for task_id in range(n_tasks):
        X_train, y_train = loader.get_data_for_task(task_id, "train")

        model, new_logits = train_with_der(
            model, X_train, y_train,
            der_buf_X, der_buf_logits, der_buf_y,
            alpha=cfg["der_alpha"], beta=cfg["der_beta"],
            epochs=cfg["n_epochs"], lr=cfg["learning_rate"],
        )

        idx = torch.randperm(len(X_train))[:mem_size]
        new_X = X_train[idx]
        new_y = y_train[idx]
        new_lg = new_logits[idx]

        if der_buf_X.numel() == 0:
            der_buf_X = new_X.clone()
            der_buf_y = new_y.clone()
            der_buf_logits = new_lg.clone()
        else:
            der_buf_X = torch.cat((der_buf_X, new_X), dim=0)
            der_buf_y = torch.cat((der_buf_y, new_y), dim=0)
            der_buf_logits = torch.cat((der_buf_logits, new_lg), dim=0)

        if der_buf_X.size(0) > mem_size:
            keep = torch.randperm(len(der_buf_X))[:mem_size]
            der_buf_X = der_buf_X[keep]
            der_buf_y = der_buf_y[keep]
            der_buf_logits = der_buf_logits[keep]

        for test_id in range(task_id + 1):
            X_test, y_test = loader.get_data_for_task(test_id, "test")
            m = test_on_task(model, X_test, y_test)
            acc[task_id, test_id] = m["acc"]
            pr[task_id, test_id] = m["pr_auc"]
    return acc, pr


METHOD_RUNNERS = {
    "naive":       _run_naive,
    "replay":      _run_replay,
    "full_replay": _run_full_replay,
    "ewc":         _run_ewc,
    "packnet":     _run_packnet,
    "si":          _run_si,
    "der":         _run_der,
}


# --------------------------------------------------------------------- #
# Main loop                                                             #
# --------------------------------------------------------------------- #

def _print_metrics(acc_m: np.ndarray, pr_m: np.ndarray, show_metrics: dict) -> None:
    avg_acc, forget_acc, avg_pr, forget_pr, _diag_pr = compute_thesis_metrics(acc_m, pr_m)
    parts = []
    if show_metrics.get("AvgPR"):     parts.append(f"AvgPR={avg_pr:.4f}")
    if show_metrics.get("ForgetPR"):  parts.append(f"ForgetPR={forget_pr:.4f}")
    if show_metrics.get("AvgAcc"):    parts.append(f"AvgAcc={avg_acc:.4f}")
    if show_metrics.get("ForgetAcc"): parts.append(f"ForgetAcc={forget_acc:.4f}")
    print("  |  ".join(parts))


def run_experiments(cfg: dict, project_root: Path) -> dict:
    """Run all configured methods on all configured datasets."""
    set_seed(cfg["seed"])

    base_dir = Path(cfg["paths"]["base_dir"])
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir
    results_dir = Path(cfg["paths"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(base_dir)
    run_axes = cfg.get("run_axes")
    run_methods = cfg["run_methods"]
    show_metrics = cfg["show_metrics"]

    all_results: dict = {}

    for ds_name, info in datasets.items():
        if run_axes is not None and info["axis"] not in run_axes:
            continue

        ds_path = str(info["path"])
        if not os.path.exists(ds_path):
            print(f"  [SKIP] {ds_name} — file not found ({ds_path})")
            continue

        print(f"\n{'='*65}")
        print(f"  DATASET: {ds_name}  |  {os.path.basename(ds_path)}")
        print(f"{'='*65}")
        t0 = time.time()

        # Fresh DataLoader per dataset -- replaces the global state reset
        # that the original notebook performed manually.
        loader = DataLoader(ds_path)
        all_results[ds_name] = {}

        for method, runner in METHOD_RUNNERS.items():
            if not run_methods.get(method, False):
                continue

            cached = load_ckpt(results_dir, ds_name, method)
            if cached is not None:
                print(f"\n  [{method}] Loaded from cache.")
                acc_m, pr_m = cached
            else:
                print(f"\n  [{method}] Running...")
                acc_m, pr_m = runner(loader, cfg)
                save_ckpt(results_dir, ds_name, method, acc_m, pr_m)

            _print_metrics(acc_m, pr_m, show_metrics)
            all_results[ds_name][method] = (acc_m, pr_m)

        print(f"\n  Completed in {time.time()-t0:.0f}s")

    print(f"\n{'='*65}")
    print(f"  Finished: {list(all_results.keys())}")
    print(f"{'='*65}")
    return all_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continual-learning multi-dataset runner.")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment.yaml"),
        help="Path to the YAML configuration file.",
    )
    p.add_argument(
        "--axes",
        nargs="*", type=int, default=None,
        help="Override 'run_axes' from the config (space-separated list).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.axes is not None:
        cfg["run_axes"] = args.axes if len(args.axes) > 0 else None

    run_experiments(cfg, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
