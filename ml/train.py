"""Train a multivariate LSTM on the windowed gold dataset.

Usage:
    python -m ml.train --gold ./data/lake/gold/weather_features

Reads gold parquet from a local path (use `mc cp --recursive
local/weather-lake/gold/ ./data/lake/gold/` to sync from MinIO first,
or point at any pyarrow-readable path).

Train/val split is time-based (no leakage). Checkpoints the model with
the lowest validation loss to `checkpoints/best.pt`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.dataset import (
    FEATURE_COLUMNS,
    WeatherWindowsDataset,
    WindowSpec,
    build_window_set,
    load_gold,
    time_based_split,
)
from ml.logging_utils import (
    EpochRecord,
    plot_loss_curves,
    setup_logger,
    write_training_log,
)
from ml.models.lstm import WeatherLSTM


def train_epoch(
    model: WeatherLSTM,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: WeatherLSTM,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weather LSTM on gold parquet.")
    parser.add_argument("--gold", required=True, help="Path to gold parquet dir/file.")
    parser.add_argument("--seq-in", type=int, default=24)
    parser.add_argument("--seq-out", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    log = setup_logger("train", ckpt_path.parent / "train.log")

    log.info(f"device={device}  loading gold from {args.gold}")
    if device.type == "cuda":
        log.info(
            f"cuda: {torch.cuda.get_device_name(0)} "
            f"(capability {torch.cuda.get_device_capability(0)}, "
            f"torch={torch.__version__}, cuda_runtime={torch.version.cuda})"
        )
    gold = load_gold(args.gold)
    n_locations = gold.groupby(["lat", "lon"]).ngroups
    log.info(f"{len(gold):,} gold rows; {n_locations} unique locations")

    spec = WindowSpec(seq_in=args.seq_in, seq_out=args.seq_out)
    X, y, anchors = build_window_set(gold, spec)
    if len(X) == 0:
        raise SystemExit(
            "[train] zero windows produced — not enough rows per location for "
            f"seq_in={args.seq_in}+seq_out={args.seq_out}. Need more gold data."
        )
    Xtr, ytr, Xva, yva = time_based_split(X, y, anchors, val_fraction=args.val_fraction)
    log.info(f"windows: train={len(Xtr)}  val={len(Xva)}")
    if len(Xva) == 0:
        raise SystemExit(
            "[train] zero validation windows — pick a smaller --val-fraction."
        )

    train_ds = WeatherWindowsDataset(Xtr, ytr)
    val_ds = WeatherWindowsDataset(
        Xva, yva, train_ds.feature_mean, train_ds.feature_std
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = WeatherLSTM(
        n_features=len(FEATURE_COLUMNS),
        seq_out=args.seq_out,
        hidden_size=args.hidden,
        num_layers=args.layers,
    ).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    records: list[EpochRecord] = []
    log.info("epoch  train_mse  val_mse")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss = eval_epoch(model, val_loader, loss_fn, device)
        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": train_ds.feature_mean,
                    "feature_std": train_ds.feature_std,
                    "feature_columns": list(FEATURE_COLUMNS),
                    "spec": asdict(spec),
                    "hyperparams": {
                        "hidden_size": args.hidden,
                        "num_layers": args.layers,
                    },
                },
                ckpt_path,
            )
        records.append(
            EpochRecord(
                epoch=epoch, train_mse=tr_loss, val_mse=va_loss, is_best=is_best
            )
        )
        marker = "  <- best" if is_best else ""
        log.info(f"{epoch:>5d}  {tr_loss:9.4f}  {va_loss:7.4f}{marker}")

    log.info(f"best val MSE: {best_val:.4f}")
    log.info(f"checkpoint saved to {ckpt_path}")

    metrics = {"best_val_mse": best_val, "epochs": args.epochs}
    metrics_path = ckpt_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info(f"metrics saved to {metrics_path}")

    csv_log_path = ckpt_path.parent / "training_log.csv"
    plot_path = ckpt_path.parent / "loss_curves.png"
    write_training_log(csv_log_path, records)
    plot_loss_curves(records, plot_path)
    log.info(f"training log saved to {csv_log_path}")
    log.info(f"loss curves saved to {plot_path}")


if __name__ == "__main__":
    main()
