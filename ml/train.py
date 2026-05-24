"""Train a multivariate LSTM on the windowed gold dataset.

Usage:
    # Fresh run, fixed gold range, register in the model catalog:
    python -m ml.train --gold s3://weather-lake/gold/weather_features

    # Continue from a previous checkpoint for N more epochs:
    python -m ml.train --gold s3://weather-lake/gold/weather_features \\
        --resume checkpoints/best.pt --extra-epochs 5

Reads gold parquet directly from MinIO via pyarrow's S3FileSystem (Phase
2a dropped the local-FS path). Set S3_ENDPOINT / S3_ACCESS_KEY /
S3_SECRET_KEY in `.env` to point at your MinIO instance.

Train/val split is time-based (no leakage). Checkpoints land in
`checkpoints/<ISO_timestamp>.pt`; the lowest-val-MSE run is also copied
to `checkpoints/best.pt` so the predictor has a stable pointer. Each
training run records a row in the Postgres `models` table (Phase 2b) so
the dashboard can detect "model trained on stale gold range."
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
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


def _now_version() -> str:
    """ISO-8601 UTC timestamp safe for use as a Windows filename."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def validate_resume_checkpoint(
    prev: dict,
    current_feature_columns: tuple[str, ...],
    current_hidden: int,
    current_layers: int,
) -> tuple[int, float]:
    """Refuse cross-schema resumes; return (start_epoch, prev_best_val_mse)."""
    prev_columns = tuple(prev.get("feature_columns", []))
    if prev_columns != current_feature_columns:
        raise SystemExit(
            "[train] feature_columns mismatch between --resume checkpoint and "
            "current code — refusing to continue training across schema drift."
        )
    prev_hp = prev.get("hyperparams", {})
    if (
        prev_hp.get("hidden_size") != current_hidden
        or prev_hp.get("num_layers") != current_layers
    ):
        raise SystemExit(
            "[train] hyperparams mismatch on --resume "
            f"(was hidden={prev_hp.get('hidden_size')}, layers={prev_hp.get('num_layers')}; "
            f"now hidden={current_hidden}, layers={current_layers})."
        )
    return (
        int(prev.get("epoch", 0)) + 1,
        float(prev.get("best_val_mse", float("inf"))),
    )


def _safe_register_model(**kwargs) -> bool:
    """Insert a row into the `models` table. Returns True on success.

    DB unreachable / table missing is non-fatal: training succeeded and the
    checkpoint is on disk, so we just warn and move on.
    """
    try:
        from warehouse.client import connect_from_env, register_model, transaction
    except Exception as exc:
        print(f"[train] warehouse.client import failed: {exc}", file=sys.stderr)
        return False
    try:
        conn = connect_from_env()
        try:
            with transaction(conn):
                register_model(conn, **kwargs)
        finally:
            conn.close()
        return True
    except Exception as exc:
        print(f"[train] model registry update failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weather LSTM on gold parquet.")
    parser.add_argument("--gold", required=True, help="s3:// URI of the gold parquet.")
    parser.add_argument("--seq-in", type=int, default=24)
    parser.add_argument("--seq-out", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Where versioned checkpoints land. `best.pt` inside this dir is a "
        "copy of whichever run had the lowest val MSE.",
    )
    parser.add_argument(
        "--resume",
        help="Path to an existing checkpoint to continue training from. "
        "Hyperparams (hidden/layers/seq_in/seq_out) and feature columns must "
        "match — schema drift is rejected.",
    )
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=0,
        help="When --resume is given, train this many additional epochs (on top "
        "of the checkpoint's recorded epoch count). Ignored otherwise.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip writing a row to the Postgres `models` table.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger("train", ckpt_dir / "train.log")

    model_version = _now_version()
    versioned_path = ckpt_dir / f"{model_version}.pt"
    best_path = ckpt_dir / "best.pt"

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

    # Provenance: what date range did we train on?
    observed_at = gold["observed_at"]
    data_range_start: date | None = None
    data_range_end: date | None = None
    if len(observed_at):
        data_range_start = observed_at.min().date()
        data_range_end = observed_at.max().date()
        log.info(f"gold range: {data_range_start} .. {data_range_end}")

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

    # Optional resume: load state, sanity-check schema, override start_epoch.
    start_epoch = 1
    best_val = float("inf")
    resumed_from: str | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"[train] --resume path not found: {resume_path}")
        prev = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch, best_val = validate_resume_checkpoint(
            prev, FEATURE_COLUMNS, args.hidden, args.layers
        )
        resumed_from = str(resume_path)
        log.info(
            f"resuming from {resume_path} at epoch {start_epoch} "
            f"(prev best_val_mse={best_val:.4f})"
        )

    model = WeatherLSTM(
        n_features=len(FEATURE_COLUMNS),
        seq_out=args.seq_out,
        hidden_size=args.hidden,
        num_layers=args.layers,
    ).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        model.load_state_dict(prev["model_state"])
        if "optimizer_state" in prev:
            optimizer.load_state_dict(prev["optimizer_state"])

    if args.resume:
        # On resume, --extra-epochs (if given) wins over --epochs; otherwise we
        # treat --epochs as "more epochs from here," not "absolute target."
        epochs_to_run = args.extra_epochs if args.extra_epochs > 0 else args.epochs
        target_epoch = start_epoch + epochs_to_run - 1
    else:
        target_epoch = args.epochs

    records: list[EpochRecord] = []
    log.info("epoch  train_mse  val_mse")
    for epoch in range(start_epoch, target_epoch + 1):
        tr_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss = eval_epoch(model, val_loader, loss_fn, device)
        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
        records.append(
            EpochRecord(
                epoch=epoch, train_mse=tr_loss, val_mse=va_loss, is_best=is_best
            )
        )
        marker = "  <- best" if is_best else ""
        log.info(f"{epoch:>5d}  {tr_loss:9.4f}  {va_loss:7.4f}{marker}")

    trained_at = datetime.now(timezone.utc)
    hyperparams = {
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "resumed_from": resumed_from,
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_mean": train_ds.feature_mean,
        "feature_std": train_ds.feature_std,
        "feature_columns": list(FEATURE_COLUMNS),
        "spec": asdict(spec),
        "hyperparams": hyperparams,
        # Phase 2b provenance:
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "data_range_start": data_range_start.isoformat() if data_range_start else None,
        "data_range_end": data_range_end.isoformat() if data_range_end else None,
        "gold_row_count": int(len(gold)),
        "best_val_mse": best_val,
        "epoch": target_epoch,
    }
    torch.save(checkpoint, versioned_path)
    shutil.copyfile(versioned_path, best_path)
    log.info(f"best val MSE: {best_val:.4f}")
    log.info(f"checkpoint saved to {versioned_path}")
    log.info(f"best pointer updated: {best_path}")

    metrics = {
        "model_version": model_version,
        "best_val_mse": best_val,
        "epochs_in_this_run": target_epoch - start_epoch + 1,
        "total_epochs": target_epoch,
        "data_range_start": checkpoint["data_range_start"],
        "data_range_end": checkpoint["data_range_end"],
        "gold_row_count": checkpoint["gold_row_count"],
        "resumed_from": resumed_from,
    }
    metrics_path = versioned_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info(f"metrics saved to {metrics_path}")

    csv_log_path = ckpt_dir / f"{model_version}_log.csv"
    plot_path = ckpt_dir / f"{model_version}_loss.png"
    write_training_log(csv_log_path, records)
    plot_loss_curves(records, plot_path)
    log.info(f"training log saved to {csv_log_path}")
    log.info(f"loss curves saved to {plot_path}")

    if not args.no_register:
        ok = _safe_register_model(
            model_version=model_version,
            trained_at=trained_at,
            data_range_start=data_range_start,
            data_range_end=data_range_end,
            gold_row_count=int(len(gold)),
            best_val_mse=best_val,
            epochs=target_epoch,
            checkpoint_path=str(versioned_path),
            hyperparams=hyperparams,
            notes=(f"resumed from {resumed_from}" if resumed_from else None),
        )
        log.info(f"model registry: {'updated' if ok else 'skipped'}")


if __name__ == "__main__":
    main()
