"""Lightweight training/eval artifacts: CSV logs + matplotlib plots.

Kept separate from train.py / evaluate.py so the training code stays
focused on training, and so the plotting can be unit-tested in isolation
if needed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Non-interactive backend — these scripts run headless (no display).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class EpochRecord:
    epoch: int
    train_mse: float
    val_mse: float
    is_best: bool


def write_training_log(path: Path, records: list[EpochRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse", "is_best"])
        for r in records:
            writer.writerow(
                [r.epoch, f"{r.train_mse:.6f}", f"{r.val_mse:.6f}", int(r.is_best)]
            )


def plot_loss_curves(records: list[EpochRecord], out_path: Path) -> None:
    epochs = [r.epoch for r in records]
    train = [r.train_mse for r in records]
    val = [r.val_mse for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train MSE", marker="o", linewidth=1.5)
    ax.plot(epochs, val, label="val MSE", marker="s", linewidth=1.5)

    best_idx = int(np.argmin(val))
    ax.scatter(
        [epochs[best_idx]],
        [val[best_idx]],
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label=f"best val (epoch {epochs[best_idx]}, MSE={val[best_idx]:.3f})",
        zorder=5,
    )

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE  (°C²)")
    ax.set_title("Training curves — multivariate LSTM, gold weather features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_per_horizon_csv(
    path: Path,
    horizons: list[int],
    mae: list[float],
    rmse: list[float],
    mape: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h_ahead", "mae_c", "rmse_c", "mape_pct"])
        for h, m, r, p in zip(horizons, mae, rmse, mape):
            writer.writerow([h, f"{m:.4f}", f"{r:.4f}", f"{p:.4f}"])


def plot_per_horizon_error(
    horizons: list[int],
    mae: list[float],
    rmse: list[float],
    persistence_mae: float,
    out_path: Path,
) -> None:
    """Grouped bars of MAE / RMSE per forecast horizon, with persistence baseline."""
    x = np.arange(len(horizons))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, mae, width, label="MAE", color="#3478f6")
    ax.bar(x + width / 2, rmse, width, label="RMSE", color="#f08c00")
    ax.axhline(
        persistence_mae,
        linestyle="--",
        color="gray",
        label=f"persistence MAE = {persistence_mae:.2f} °C",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"h+{h}" for h in horizons])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("error  (°C)")
    ax.set_title("Per-horizon error vs. persistence baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_predictions_vs_actual(
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    n_samples: int = 6,
) -> None:
    """Overlay model forecasts on the ground-truth horizon for a few sample windows."""
    n = min(n_samples, len(pred))
    if n == 0:
        return
    # Evenly-spaced indices across the hold-out set so we sample old + new.
    idx = np.linspace(0, len(pred) - 1, n, dtype=int)
    seq_out = pred.shape[1]
    h = np.arange(1, seq_out + 1)

    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes_flat, idx):
        ax.plot(h, true[i], marker="o", label="actual", color="#1a7f37")
        ax.plot(
            h, pred[i], marker="s", label="forecast", color="#cf222e", linestyle="--"
        )
        ax.set_title(f"hold-out window #{i}")
        ax.set_xlabel("hours ahead")
        ax.grid(True, alpha=0.3)
    axes_flat[0].set_ylabel("temperature_2m  (°C)")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(
        f"Predictions vs. actuals — {n} sample windows from hold-out set", y=1.02
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
