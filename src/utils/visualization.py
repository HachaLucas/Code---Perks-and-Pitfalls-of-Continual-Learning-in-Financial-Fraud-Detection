"""
Visualisation utilities for the per-axis thesis figures.

These functions reproduce the visualisation cells from the original
notebook. They consume the saved checkpoints in ``results_dir`` and write
PNGs to ``figures_dir``.

Public API:

* ``METHOD_STYLE`` / ``METRICS`` -- styling constants reused everywhere.
* ``load_results_from_disk(results_dir)`` -> dict[ds_name -> {method: (acc, pr)}]
* ``build_axis_groups(ds_results, datasets_dict)`` -> dict[axis -> list[(ds, info, mdict)]]
* ``plot_per_axis(axis_groups, figures_dir)`` -- per-axis bar/line charts.
* ``plot_per_axis_heatmaps(axis_groups, figures_dir)`` -- per-axis heatmaps.
* ``plot_per_dataset_pr_heatmaps(ds_results, figures_dir)`` -- per-(ds, method) PR-AUC heatmaps.
* ``plot_per_dataset_forget_pr_heatmaps(ds_results, figures_dir)`` -- per-(ds, method) ForgetPR heatmaps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .metrics import compute_thesis_metrics


# Method styling -- add/reorder here; only methods with results get plotted.
METHOD_STYLE: Dict[str, Dict[str, str]] = {
    "naive":       {"color": "#4C72B0", "label": "Naive"},
    "replay":      {"color": "#55A868", "label": "Replay"},
    "full_replay": {"color": "#DD8452", "label": "Full Replay"},
    "ewc":         {"color": "#C44E52", "label": "EWC"},
    "packnet":     {"color": "#CCB974", "label": "PackNet"},
    "si":          {"color": "#8172B2", "label": "SI"},
    "der":         {"color": "#64B5CD", "label": "DER++"},
}

def _format_ds_label(ds_name: str) -> str:
    """Convert a raw dataset key to a human-readable title.

    axis1_delta1.5_sudden      -> "Sudden: Delta = 1.5"
    axis2_delta0.3_incremental -> "Incremental: Delta = 0.3"
    axis3_freeze_k3            -> "Freeze: k = 3"
    axis4_offset2.0            -> "Offset: Gamma = 2.0"
    axis5_pattern_rotation     -> "Pattern Rotation"
    axis0_baseline             -> "Baseline"
    """
    name = re.sub(r"^axis\d+_", "", ds_name)
    m = re.match(r"delta([\d.]+)_sudden", name)
    if m:
        return f"Sudden: Delta = {m.group(1)}"
    m = re.match(r"delta([\d.]+)_incremental", name)
    if m:
        return f"Incremental: Delta = {m.group(1)}"
    m = re.match(r"freeze_k(\d+)", name)
    if m:
        return f"Freeze: k = {m.group(1)}"
    m = re.match(r"offset([\d.]+)", name)
    if m:
        return f"Offset: Gamma = {m.group(1)}"
    if name == "pattern_rotation":
        return "Pattern Rotation"
    if name == "baseline":
        return "Baseline"
    return name.replace("_", " ").title()


# (label, index in the compute_thesis_metrics tuple)
METRICS: List[Tuple[str, int]] = [
    ("AvgPR", 2),
    ("ForgetPR", 3),
    ("DiagPR", 4),
]


# --------------------------------------------------------------------- #
# Disk helpers                                                          #
# --------------------------------------------------------------------- #

def load_results_from_disk(results_dir: str | Path) -> dict:
    """
    Walk ``results_dir`` and return a nested dict::

        {dataset_name: {method_name: (acc_mat, pr_mat)}}

    Only datasets with at least one (acc, pr) pair are included.
    """
    results_dir = Path(results_dir)
    ds_results: dict = {}
    if not results_dir.exists():
        return ds_results

    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        acc_methods = {f.stem.replace("_acc", "") for f in ds_dir.glob("*_acc.npy")}
        pr_methods = {f.stem.replace("_pr", "") for f in ds_dir.glob("*_pr.npy")}
        methods = sorted(acc_methods & pr_methods)
        if not methods:
            continue
        ds_results[ds_dir.name] = {
            m: (np.load(ds_dir / f"{m}_acc.npy"), np.load(ds_dir / f"{m}_pr.npy"))
            for m in methods
        }
    return ds_results


def build_axis_groups(ds_results: dict, datasets_dict: dict) -> dict:
    """
    Group ``ds_results`` by their axis (as defined in ``datasets_dict``).

    Returns ``{axis_id: [(ds_name, info, mdict), ...]}``.
    """
    axis_groups: dict = {}
    for ds_name, info in datasets_dict.items():
        if ds_name not in ds_results:
            continue
        axis_groups.setdefault(info["axis"], []).append(
            (ds_name, info, ds_results[ds_name])
        )
    return axis_groups


# --------------------------------------------------------------------- #
# Internal helpers                                                      #
# --------------------------------------------------------------------- #

def _axis_methods(entries: Iterable) -> List[str]:
    """Return methods ordered by METHOD_STYLE that are present in any dataset."""
    present: set = set()
    for _, _, mdict in entries:
        present |= mdict.keys()
    return [m for m in METHOD_STYLE if m in present]


def _save(fig, figures_dir: Path, name: str) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")


def _auto_ylim(ax, pad_frac: float = 0.15, min_pad: float = 0.005) -> None:
    """Set y-axis to actual data range + small margin instead of fixed 0-1."""
    ydata = []
    for line in ax.get_lines():
        ydata.extend([float(v) for v in line.get_ydata() if not (v != v)])  # skip NaN
    for patch in ax.patches:
        h = patch.get_height()
        if h == h:  # not NaN
            ydata.append(h)
    if not ydata:
        return
    ymin, ymax = min(ydata), max(ydata)
    span = ymax - ymin
    pad = max(span * pad_frac, min_pad)
    ax.set_ylim(ymin - pad, ymax + pad)


# --------------------------------------------------------------------- #
# Public plotting functions                                             #
# --------------------------------------------------------------------- #

def plot_per_axis(axis_groups: dict, figures_dir: str | Path) -> None:
    """Per-axis bar/line plots covering AvgPR, ForgetPR, DiagPR."""
    figures_dir = Path(figures_dir)

    if not axis_groups:
        print("Results folder exists but no dataset names matched DATASETS.")
        return

    # AXIS 0: Baseline -- bar chart
    if 0 in axis_groups:
        entries = axis_groups[0]
        methods = _axis_methods(entries)
        _, _, mdict = entries[0]  # only 1 dataset
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Baseline", fontsize=13, fontweight="bold")
        for ax, (metric_label, midx) in zip(axes, METRICS[:2]):
            labels, vals, colors = [], [], []
            for m in methods:
                if m not in mdict:
                    continue
                met = compute_thesis_metrics(*mdict[m])
                labels.append(METHOD_STYLE.get(m, {}).get("label", m))
                vals.append(met[midx])
                colors.append(METHOD_STYLE.get(m, {}).get("color", "gray"))
            bars = ax.bar(labels, vals, color=colors)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            ax.set_title(metric_label)
            ax.set_ylabel(metric_label)
            _auto_ylim(ax)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        _save(fig, figures_dir, "axis0_baseline.png")
        plt.close(fig)

    # AXIS 1: Drift Magnitude Sudden -- line plot (x = scale)
    if 1 in axis_groups:
        entries = sorted(axis_groups[1], key=lambda t: t[1]["scale"])
        methods = _axis_methods(entries)
        for metric_label, midx in METRICS:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f"Drift Magnitude | {metric_label}", fontweight="bold")
            for m in methods:
                xs, ys = [], []
                for _, info, mdict in entries:
                    if m not in mdict:
                        continue
                    xs.append(info["scale"])
                    ys.append(compute_thesis_metrics(*mdict[m])[midx])
                if xs:
                    style = METHOD_STYLE.get(m, {})
                    ax.plot(xs, ys, marker="o", color=style.get("color"),
                            label=style.get("label", m))
            ax.set_xlabel("Drift Magnitude (scale factor)")
            ax.set_ylabel(metric_label)
            _auto_ylim(ax)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save(fig, figures_dir, f"axis1_drift_magnitude_{metric_label}.png")
            plt.close(fig)

    # AXIS 2: Incremental Drift -- line plot (x = delta)
    if 2 in axis_groups:
        entries = axis_groups[2]
        methods = _axis_methods(entries)
        for metric_label, midx in METRICS:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f"Incremental Drift | {metric_label}", fontweight="bold")
            for m in methods:
                xs, ys = [], []
                for _, info, mdict in sorted(entries, key=lambda t: t[1]["scale"]):
                    if m not in mdict:
                        continue
                    xs.append(info["scale"])
                    ys.append(compute_thesis_metrics(*mdict[m])[midx])
                if xs:
                    style = METHOD_STYLE.get(m, {})
                    ax.plot(xs, ys, marker="o", color=style.get("color"),
                            label=style.get("label", m))
            ax.set_xlabel("Drift Magnitude (delta)")
            ax.set_ylabel(metric_label)
            _auto_ylim(ax)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save(fig, figures_dir, f"axis2_incremental_drift_{metric_label}.png")
            plt.close(fig)

    # AXIS 3: Freeze Duration -- line plot (x = k)
    if 3 in axis_groups:
        entries = sorted(axis_groups[3], key=lambda t: t[1]["k"])
        methods = _axis_methods(entries)
        for metric_label, midx in METRICS:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f"Freeze Duration | {metric_label}", fontweight="bold")
            for m in methods:
                xs, ys = [], []
                for _, info, mdict in entries:
                    if m not in mdict:
                        continue
                    xs.append(info["k"])
                    ys.append(compute_thesis_metrics(*mdict[m])[midx])
                if xs:
                    style = METHOD_STYLE.get(m, {})
                    ax.plot(xs, ys, marker="o", color=style.get("color"),
                            label=style.get("label", m))
            ax.set_xlabel("Frozen Periods (k)")
            ax.set_ylabel(metric_label)
            _auto_ylim(ax)
            ax.set_xticks(sorted({t[1]["k"] for t in entries}))
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save(fig, figures_dir, f"axis3_freeze_duration_{metric_label}.png")
            plt.close(fig)

    # AXIS 4: Offset -- line plot (x = offset)
    if 4 in axis_groups:
        entries = sorted(axis_groups[4], key=lambda t: t[1]["offset"])
        methods = _axis_methods(entries)
        for metric_label, midx in METRICS:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f"Offset | {metric_label}", fontweight="bold")
            for m in methods:
                xs, ys = [], []
                for _, info, mdict in entries:
                    if m not in mdict:
                        continue
                    xs.append(info["offset"])
                    ys.append(compute_thesis_metrics(*mdict[m])[midx])
                if xs:
                    style = METHOD_STYLE.get(m, {})
                    ax.plot(xs, ys, marker="o", color=style.get("color"),
                            label=style.get("label", m))
            ax.set_xlabel("Offset")
            ax.set_ylabel(metric_label)
            _auto_ylim(ax)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save(fig, figures_dir, f"axis4_offset_{metric_label}.png")
            plt.close(fig)

    # AXIS 5: Pattern Rotation -- bar chart
    if 5 in axis_groups:
        entries = axis_groups[5]
        methods = _axis_methods(entries)
        _, _, mdict = entries[0]  # only 1 dataset
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Pattern Rotation", fontsize=13, fontweight="bold")
        for ax, (metric_label, midx) in zip(axes, METRICS[:2]):
            labels, vals, colors = [], [], []
            for m in methods:
                if m not in mdict:
                    continue
                met = compute_thesis_metrics(*mdict[m])
                labels.append(METHOD_STYLE.get(m, {}).get("label", m))
                vals.append(met[midx])
                colors.append(METHOD_STYLE.get(m, {}).get("color", "gray"))
            bars = ax.bar(labels, vals, color=colors)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            ax.set_title(metric_label)
            ax.set_ylabel(metric_label)
            _auto_ylim(ax)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        _save(fig, figures_dir, "axis5_pattern_rotation.png")
        plt.close(fig)

    print("\nAll per-axis figures generated.")


def plot_per_axis_heatmaps(axis_groups: dict, figures_dir: str | Path) -> None:
    """Per-axis cross-dataset heatmaps for each metric."""
    figures_dir = Path(figures_dir)

    AXIS_CONFIG = {
        1: {"title": "Drift Magnitude", "sort_key": lambda t: t[1]["scale"],  "x_label": "Drift Magnitude (scale)"},
        2: {"title": "Incremental Drift",   "sort_key": lambda t: t[1]["scale"],  "x_label": "Drift Magnitude (delta)"},
        3: {"title": "Freeze Duration", "sort_key": lambda t: t[1]["k"],      "x_label": "Frozen Periods (k)"},
        4: {"title": "Offset",          "sort_key": lambda t: t[1]["offset"], "x_label": "Offset"},
        5: {"title": "Pattern Rotation","sort_key": lambda t: t[1]["scale"],  "x_label": "Rotation Scale"},
    }

    for ax_idx, cfg in AXIS_CONFIG.items():
        if ax_idx not in axis_groups:
            continue

        entries = sorted(axis_groups[ax_idx], key=cfg["sort_key"])
        methods = _axis_methods(entries)
        method_labels = [METHOD_STYLE[m]["label"] for m in methods]
        x_labels = [info["label"] for _, info, _ in entries]

        for metric_label, midx in METRICS:
            mat = np.full((len(methods), len(entries)), np.nan)
            for ci, (_, _, mdict) in enumerate(entries):
                for ri, m in enumerate(methods):
                    if m in mdict:
                        mat[ri, ci] = compute_thesis_metrics(*mdict[m])[midx]

            vmin, vmax = np.nanmin(mat), np.nanmax(mat)

            if metric_label == "ForgetPR":
                abs_max = max(abs(vmin), abs(vmax))
                vmin_plot, vmax_plot = -abs_max, abs_max
                cmap = "RdYlGn_r"  # reversed: red = forgetting (positive), green = backward transfer (negative)
            else:
                vmin_plot, vmax_plot = vmin, vmax
                cmap = "RdYlGn"

            fig_w = max(8, len(x_labels) * 1.0)
            fig_h = max(3, len(methods) * 0.65)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_title(f"{cfg['title']} | {metric_label}", fontweight="bold")

            im = ax.imshow(mat, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, aspect="auto")

            for ri in range(len(methods)):
                for ci in range(len(x_labels)):
                    val = mat[ri, ci]
                    if not np.isnan(val):
                        norm_val = (
                            (val - vmin_plot) / (vmax_plot - vmin_plot)
                            if vmax_plot != vmin_plot else 0.5
                        )
                        txt_col = "black" if 0.25 < norm_val < 0.75 else "white"
                        ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                                fontsize=8, fontweight="bold", color=txt_col)

            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(method_labels, fontsize=10)
            ax.set_xlabel(cfg["x_label"])
            plt.colorbar(im, ax=ax, label=metric_label, fraction=0.03, pad=0.04)
            plt.tight_layout()
            _save(fig, figures_dir, f"axis{ax_idx}_heatmap_{metric_label}.png")
            plt.close(fig)

    print("All per-axis heatmaps generated.")


def plot_method_axis_heatmap(axis_groups: dict, figures_dir: str | Path) -> None:
    """7-method × 5-axis summary heatmap (axes 1-5, axis 0 excluded).

    Each cell = mean of the metric over all datasets within that axis.
    Produces one figure for AvgPR and one for ForgetPR.
    """
    figures_dir = Path(figures_dir)

    AXIS_LABELS = {
        1: "Drift Mag.\n(Sudden)",
        2: "Drift Mag.\n(Incremental)",
        3: "Freeze\nDuration",
        4: "Offset",
        5: "Pattern\nRotation",
    }

    axes_present = sorted(a for a in axis_groups if a in AXIS_LABELS)
    if not axes_present:
        print("No axes 1-5 found in axis_groups.")
        return

    METHOD_ORDER = ["naive", "ewc", "si", "replay", "full_replay", "der", "packnet"]
    all_methods = [
        m for m in METHOD_ORDER
        if any(
            m in mdict
            for ax_idx in axes_present
            for _, _, mdict in axis_groups[ax_idx]
        )
    ]
    method_labels = [METHOD_STYLE[m]["label"] for m in all_methods]
    x_labels = [AXIS_LABELS[a] for a in axes_present]

    for metric_label, midx in METRICS[:2]:  # AvgPR and ForgetPR only
        mat = np.full((len(all_methods), len(axes_present)), np.nan)

        for ci, ax_idx in enumerate(axes_present):
            for ri, m in enumerate(all_methods):
                vals = [
                    compute_thesis_metrics(*mdict[m])[midx]
                    for _, _, mdict in axis_groups[ax_idx]
                    if m in mdict
                ]
                if vals:
                    mat[ri, ci] = float(np.mean(vals))

        vmin, vmax = np.nanmin(mat), np.nanmax(mat)
        if metric_label == "ForgetPR":
            abs_max = max(abs(vmin), abs(vmax), 1e-6)
            vmin_plot, vmax_plot = -abs_max, abs_max
            cmap = "RdYlGn_r"  # reversed: red = forgetting (positive)
        else:
            vmin_plot, vmax_plot = vmin, vmax
            cmap = "RdYlGn"

        fig, ax = plt.subplots(figsize=(max(8, len(axes_present) * 1.8), max(4, len(all_methods) * 0.85)))
        ax.set_title(
            f"Method × Drift Scenario — {metric_label}  (mean over datasets per drift scenario)",
            fontweight="bold", pad=12,
        )

        im = ax.imshow(mat, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, aspect="auto")

        for ri in range(len(all_methods)):
            for ci in range(len(axes_present)):
                val = mat[ri, ci]
                if not np.isnan(val):
                    norm_val = (
                        (val - vmin_plot) / (vmax_plot - vmin_plot)
                        if vmax_plot != vmin_plot else 0.5
                    )
                    txt_col = "black" if 0.25 < norm_val < 0.75 else "white"
                    ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=txt_col)

        ax.set_xticks(range(len(axes_present)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_yticks(range(len(all_methods)))
        ax.set_yticklabels(method_labels, fontsize=11)
        ax.set_xlabel("Drift Axis", fontsize=11)

        cbar_label = metric_label + ("  (+ = forgetting)" if metric_label == "ForgetPR" else "")
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.03, pad=0.04)
        plt.tight_layout()
        _save(fig, figures_dir, f"method_axis_heatmap_{metric_label}.png")
        plt.close(fig)

    print("Method × axis heatmaps generated.")


def plot_per_dataset_pr_heatmaps(ds_results: dict, figures_dir: str | Path) -> None:
    """Per-(dataset, method) PR-AUC matrix heatmaps."""
    figures_dir = Path(figures_dir)
    method_order = list(METHOD_STYLE.keys())

    for ds_name, methods_data in ds_results.items():
        available = [m for m in method_order if m in methods_data]
        if not available:
            continue

        ds_dir = figures_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        for method in available:
            _, pr_mat = methods_data[method]
            n = pr_mat.shape[0]
            task_labels = [f"T{t}" for t in range(1, n + 1)]

            masked = np.where(pr_mat == 0, np.nan, pr_mat)
            style = METHOD_STYLE.get(method, {"color": "gray", "label": method})

            fig, ax = plt.subplots(figsize=(10, 9))
            fig.suptitle(
                f"PR-AUC Heatmap — {_format_ds_label(ds_name)}\n"
                "Row = after training on task | Column = tested on task | "
                "Diagonal = peak performance",
                fontsize=13, fontweight="bold",
            )
            ax.set_title(style["label"], fontsize=13, fontweight="bold",
                         color=style["color"], pad=12)

            im = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

            for row in range(n):
                for col in range(n):
                    val = pr_mat[row, col]
                    if val > 0:
                        ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color="black", fontweight="bold")

            for t in range(n):
                ax.add_patch(plt.Rectangle(
                    (t - 0.5, t - 0.5), 1, 1,
                    fill=False, edgecolor="steelblue", linewidth=2.5,
                ))

            ax.set_xlabel("Test Task", fontsize=11)
            ax.set_ylabel("After Training On Task", fontsize=11)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(task_labels, fontsize=9)
            ax.set_yticklabels(task_labels, fontsize=9)
            plt.colorbar(im, ax=ax, label="PR-AUC", fraction=0.046, pad=0.04)

            plt.tight_layout()
            _save(fig, figures_dir, f"{ds_name}/{method}_pr_heatmap.png")
            plt.close(fig)


def plot_per_dataset_forget_pr_heatmaps(ds_results: dict, figures_dir: str | Path) -> None:
    """Per-(dataset, method) ForgetPR heatmaps.

    forget_mat[i, j] = pr_mat[j, j] - pr_mat[i, j]  for i > j (lower triangle).
    Positive = forgetting, negative = backward transfer.
    """
    figures_dir = Path(figures_dir)
    method_order = list(METHOD_STYLE.keys())

    for ds_name, methods_data in ds_results.items():
        available = [m for m in method_order if m in methods_data]
        if not available:
            continue

        ds_dir = figures_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        for method in available:
            _, pr_mat = methods_data[method]
            n = pr_mat.shape[0]
            task_labels = [f"T{t}" for t in range(1, n + 1)]

            forget_mat = np.full((n, n), np.nan)
            for i in range(1, n):
                for j in range(i):
                    diag_val = pr_mat[j, j]
                    if diag_val > 0 and pr_mat[i, j] > 0:
                        forget_mat[i, j] = diag_val - pr_mat[i, j]

            abs_max = np.nanmax(np.abs(forget_mat)) if not np.all(np.isnan(forget_mat)) else 1.0
            style = METHOD_STYLE.get(method, {"color": "gray", "label": method})

            fig, ax = plt.subplots(figsize=(10, 9))
            fig.suptitle(
                f"ForgetPR Heatmap — {_format_ds_label(ds_name)}\n"
                "Row = after training on task | Column = earlier task | "
                "Positive = forgetting, Negative = backward transfer",
                fontsize=13, fontweight="bold",
            )
            ax.set_title(style["label"], fontsize=13, fontweight="bold",
                         color=style["color"], pad=12)

            im = ax.imshow(forget_mat, vmin=-abs_max, vmax=abs_max, cmap="RdYlGn_r", aspect="auto")

            for i in range(1, n):
                for j in range(i):
                    val = forget_mat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                fontsize=8, color="black", fontweight="bold")

            ax.set_xlabel("Earlier Task (peak reference)", fontsize=11)
            ax.set_ylabel("After Training On Task", fontsize=11)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(task_labels, fontsize=9)
            ax.set_yticklabels(task_labels, fontsize=9)
            plt.colorbar(im, ax=ax, label="ForgetPR (peak − final)", fraction=0.046, pad=0.04)

            plt.tight_layout()
            _save(fig, figures_dir, f"{ds_name}/{method}_forget_pr_heatmap.png")
            plt.close(fig)


def plot_bundled_appendix_heatmaps(
    axis_groups: dict,
    ds_results: dict,  # noqa: ARG001 — kept for API consistency with other plot functions
    figures_dir: str | Path,
) -> None:
    """One compact figure per axis: grid of PR-AUC heatmaps (rows=datasets, cols=methods).

    Intended for the thesis appendix so all per-dataset heatmaps of an axis
    fit on a single page without individual tick labels cluttering the view.
    Saves ``appendix_axis{N}_bundled_pr.png`` for every axis present in
    ``axis_groups``.
    """
    figures_dir = Path(figures_dir)

    # Sort key for each axis; None means keep whatever order build_axis_groups gave.
    SORT_KEYS: dict = {
        0: None,
        1: lambda t: t[1]["scale"],
        2: lambda t: t[1]["scale"],
        3: lambda t: t[1]["k"],
        4: lambda t: t[1]["offset"],
        5: lambda t: t[1]["scale"],
    }

    for axis_id, entries in sorted(axis_groups.items()):
        sort_key = SORT_KEYS.get(axis_id)
        sorted_entries = sorted(entries, key=sort_key) if sort_key else list(entries)

        methods = _axis_methods(sorted_entries)
        if not methods:
            continue

        n_rows = len(sorted_entries)
        n_cols = len(methods)

        # Reserve ~0.8 in on the left for row labels, ~0.9 in on the right for colorbar.
        fig_w = n_cols * 1.5 + 0.8 + 0.9
        fig_h = n_rows * 1.3 + 0.5  # extra for column header row

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
        fig.subplots_adjust(wspace=0.05, hspace=0.15)

        # Column headers — method names above the top row.
        for ci, m in enumerate(methods):
            axes[0, ci].set_title(
                METHOD_STYLE[m]["label"], fontsize=8, fontweight="bold", pad=3
            )

        last_im = None
        for ri, (_, info, mdict) in enumerate(sorted_entries):
            # Dataset label on the left of the first column.
            axes[ri, 0].set_ylabel(
                info["label"], fontsize=7, rotation=0,
                ha="right", va="center", labelpad=4,
            )

            for ci, m in enumerate(methods):
                ax = axes[ri, ci]
                ax.set_xticks([])
                ax.set_yticks([])

                if m not in mdict:
                    # Empty cell — keep axes visible so grid stays regular.
                    ax.set_facecolor("#e8e8e8")
                    continue

                _, pr_mat = mdict[m]
                masked = np.where(pr_mat == 0, np.nan, pr_mat)
                last_im = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

        if last_im is not None:
            cbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                shrink=0.8,
            )
            cbar.set_label("PR-AUC", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        figures_dir.mkdir(parents=True, exist_ok=True)
        out_path = figures_dir / f"appendix_axis{axis_id}_bundled_pr.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)

    print("All bundled appendix heatmaps generated.")
