"""Backtest the saved checkpoint on the time-based hold-out and report
MAE / RMSE / MAPE — both pooled across all forecast horizons and broken
out per horizon hour, since 1-hour-ahead is much easier than 6-hours-ahead.

Usage:
    python -m ml.evaluate --gold ./data/lake/gold/weather_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.dataset import (
    WeatherWindowsDataset,
    WindowSpec,
    build_window_set,
    load_gold,
    time_based_split,
)
from ml.models.lstm import WeatherLSTM


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-3) -> float:
    return float(
        np.mean(np.abs((pred - true) / np.where(np.abs(true) < eps, eps, true))) * 100
    )


@torch.no_grad()
def collect_predictions(
    model: WeatherLSTM, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the saved checkpoint.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    feature_cols = tuple(ckpt["feature_columns"])
    spec = WindowSpec(**ckpt["spec"])
    hp = ckpt["hyperparams"]

    print(f"[eval] loading gold from {args.gold}")
    gold = load_gold(args.gold)
    X, y, anchors = build_window_set(gold, spec, feature_cols=feature_cols)
    _, _, Xva, yva = time_based_split(X, y, anchors, val_fraction=args.val_fraction)
    if len(Xva) == 0:
        raise SystemExit("[eval] zero validation windows.")

    val_ds = WeatherWindowsDataset(Xva, yva, ckpt["feature_mean"], ckpt["feature_std"])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = WeatherLSTM(
        n_features=len(feature_cols),
        seq_out=spec.seq_out,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    pred, true = collect_predictions(model, val_loader, device)
    print(f"[eval] {len(pred)} hold-out windows, seq_out={spec.seq_out}")

    print("\nPooled across all horizons:")
    print(f"  MAE  = {_mae(pred, true):6.3f} °C")
    print(f"  RMSE = {_rmse(pred, true):6.3f} °C")
    print(f"  MAPE = {_mape(pred, true):6.2f} %")

    print("\nPer-horizon (hours ahead):")
    print("  h+  MAE  RMSE  MAPE%")
    for h in range(spec.seq_out):
        mae = _mae(pred[:, h], true[:, h])
        rmse = _rmse(pred[:, h], true[:, h])
        mape = _mape(pred[:, h], true[:, h])
        print(f"  {h + 1:>2d}  {mae:5.2f} {rmse:5.2f} {mape:5.1f}")

    # Persistence baseline: "next L_out values = last observed value".
    # Xva is the *un-normalized* raw window array (normalization happens
    # inside WeatherWindowsDataset), and feature index 0 is temperature_2m,
    # so the last input row's column 0 is the most recent observed temp
    # in original units — no un-scaling needed.
    target_idx = feature_cols.index("temperature_2m")
    last_obs = Xva[:, -1, target_idx]
    persistence = np.broadcast_to(last_obs[:, None], yva.shape)
    print("\nPersistence baseline (last-observed-value):")
    print(f"  MAE  = {_mae(persistence, yva):6.3f} °C")
    print(f"  RMSE = {_rmse(persistence, yva):6.3f} °C")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
