"""Daily walk-forward backtest — one model, many anchors, dual purpose.

Run as an Airflow task after `train_model` and `load_warehouse`:

  1. Load `checkpoints/best.pt` (trained on gold[<= T - cutoff_days] so the
     last `cutoff_days` of gold are an honest holdout the model never saw).
  2. For each cell, enumerate anchors in [T - lookback_days, T] at stride
     `stride_hours` (default: one anchor per day → 8 anchors per cell over 7
     days). At each anchor, predict the next `seq_out` hours.
  3. Persist all predictions under one model_version: `walkforward:<orig>`.
  4. Backfill `actual_value` for anchors whose target_time is already past.
  5. Upsert into `backtest_groups` so the dashboard's BT table picks them up.

The rightmost anchor (T) is the operational forecast — its targets extend
into the future, so `actual_value` stays NULL until subsequent DAG runs
backfill from new observations.

Usage:
    python -m ml.walk_forward [--config config/train.yaml]
        [--checkpoint PATH] [--gold s3://...]
        [--lookback-days N] [--stride-hours N]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.config import load_config
from ml.dataset import WindowSpec, load_gold
from ml.models.lstm import WeatherLSTM
from warehouse.backtest_groups import UPSERT_SQL as BACKTEST_GROUPS_UPSERT_SQL
from warehouse.client import (
    PredictionRow,
    backfill_actuals_for_version,
    connect_from_env,
    insert_predictions,
    list_location_ids,
    transaction,
)


# ---------- Pure helpers (anchor math + batching + checkpoint I/O) ----------


def enumerate_anchors(n_rows: int, seq_in: int, stride_hours: int = 1) -> list[int]:
    """Indices into a per-cell series where a full input window is available.

    First valid anchor is `seq_in - 1` (need `seq_in` rows ending here);
    last is `n_rows - 1` (its targets extend into the future and will have
    no actuals — the caller handles that). Returns empty if not enough rows.
    """
    if n_rows < seq_in:
        return []
    return list(range(seq_in - 1, n_rows, max(1, stride_hours)))


def target_times_for(anchor_ts: datetime, seq_out: int) -> list[datetime]:
    """target_time[k-1] = anchor + k hours, for k in 1..seq_out."""
    return [anchor_ts + timedelta(hours=k) for k in range(1, seq_out + 1)]


def build_inference_batch(
    feats_norm: np.ndarray, anchors: list[int], seq_in: int
) -> np.ndarray:
    """Stack one (seq_in, n_features) window per anchor → (n_anchors, ...)."""
    return np.stack([feats_norm[a - seq_in + 1 : a + 1] for a in anchors]).astype(
        np.float32
    )


def _to_utc_dt(ts) -> datetime:
    """Normalize pandas/numpy timestamps to tz-aware UTC datetime."""
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def backtest_one_cell(
    *,
    cell_rows: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    model: WeatherLSTM,
    device: torch.device,
    spec: WindowSpec,
    stride_hours: int,
    anchor_start: pd.Timestamp | None,
    anchor_end: pd.Timestamp | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (anchor_ts: shape (n,), predictions: shape (n, seq_out))."""
    if len(cell_rows) == 0:
        return (
            np.empty(0, dtype="datetime64[ns]"),
            np.empty((0, spec.seq_out), dtype=np.float32),
        )

    feats = cell_rows[feature_columns].to_numpy(dtype=np.float32)
    feats_norm = (feats - feature_mean) / feature_std
    ts = cell_rows["observed_at"].to_numpy()

    candidates = enumerate_anchors(len(cell_rows), spec.seq_in, stride_hours)
    if anchor_start is not None or anchor_end is not None:
        lo = np.datetime64(anchor_start) if anchor_start is not None else None
        hi = np.datetime64(anchor_end) if anchor_end is not None else None
        candidates = [
            i
            for i in candidates
            if (lo is None or ts[i] >= lo) and (hi is None or ts[i] <= hi)
        ]
    if not candidates:
        return (
            np.empty(0, dtype=ts.dtype),
            np.empty((0, spec.seq_out), dtype=np.float32),
        )

    batch = build_inference_batch(feats_norm, candidates, spec.seq_in)
    with torch.no_grad():
        x_t = torch.from_numpy(batch).to(device)
        preds = model(x_t).cpu().numpy()

    return ts[candidates], preds.astype(np.float32)


def _load_checkpoint(path: Path, device: torch.device):
    """Returns (model, feature_columns, spec, feature_mean, feature_std, orig_version)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    feature_columns = list(ckpt["feature_columns"])
    spec = WindowSpec(**ckpt["spec"])
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    hp = ckpt["hyperparams"]
    orig_version = ckpt.get("model_version") or "unknown"

    model = WeatherLSTM(
        n_features=len(feature_columns),
        seq_out=spec.seq_out,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feature_columns, spec, feature_mean, feature_std, orig_version


def _persist_cell(
    *,
    conn,
    model_version: str,
    location_id: int,
    anchor_ts: np.ndarray,
    preds: np.ndarray,
    seq_out: int,
    seq_in: int,
) -> int:
    """Insert one batch of walk-forward rows. Returns total rows written."""
    written = 0
    with transaction(conn):
        for i in range(len(anchor_ts)):
            pma = _to_utc_dt(anchor_ts[i])
            rows = [
                PredictionRow(
                    target_time=pma + timedelta(hours=k + 1),
                    predicted_value=float(preds[i, k]),
                )
                for k in range(seq_out)
            ]
            written += insert_predictions(
                conn,
                model_version=model_version,
                location_id=location_id,
                prediction_made_at=pma,
                rows=rows,
                seq_in=seq_in,
            )
    return written


# ---------- Daily DAG entry point ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily walk-forward backtest (operational forecast + "
        "lookback-days evaluation in one pass)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained checkpoint (default: <checkpoint_dir>/best.pt).",
    )
    parser.add_argument("--gold", default=None, help="s3:// URI of the gold parquet.")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--stride-hours", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    gold_path = args.gold if args.gold is not None else cfg.paths.gold
    lookback_days = (
        args.lookback_days
        if args.lookback_days is not None
        else cfg.backtest.lookback_days
    )
    stride_hours = (
        args.stride_hours
        if args.stride_hours is not None
        else cfg.backtest.stride_hours
    )
    ckpt_path = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(cfg.paths.checkpoint_dir) / "best.pt"
    )

    if not ckpt_path.exists():
        raise SystemExit(f"[walk_forward] checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[walk_forward] device={device}  checkpoint={ckpt_path}")

    model, feature_columns, spec, feature_mean, feature_std, orig_version = (
        _load_checkpoint(ckpt_path, device)
    )
    if orig_version == "unknown":
        raise SystemExit(
            "[walk_forward] checkpoint has no model_version; refusing to write "
            "predictions without a traceable identity."
        )

    # One model_version per DAG run (= per checkpoint). Re-running walk_forward
    # against the same checkpoint is a no-op via the UNIQUE constraint on
    # (model_version, location_id, prediction_made_at, target_time). A fresh
    # training produces a new orig_version → fresh walkforward rows.
    walkforward_version = f"walkforward:{orig_version}"
    print(
        f"[walk_forward] orig_version={orig_version} "
        f"seq_in={spec.seq_in} seq_out={spec.seq_out}"
    )
    print(f"[walk_forward] writing model_version={walkforward_version}")

    print(f"[walk_forward] loading gold from {gold_path}")
    gold = load_gold(gold_path)
    if gold.empty:
        raise SystemExit("[walk_forward] gold is empty; nothing to do.")

    t_max = pd.Timestamp(gold["observed_at"].max())
    anchor_lo = t_max - pd.Timedelta(days=lookback_days)
    print(
        f"[walk_forward] anchor range: [{anchor_lo} .. {t_max}] "
        f"(stride={stride_hours}h, lookback={lookback_days}d)"
    )

    cells = sorted(
        {(float(lat), float(lon)) for lat, lon in gold[["lat", "lon"]].to_numpy()}
    )

    try:
        conn = connect_from_env()
        try:
            location_ids = list_location_ids(conn)
        finally:
            conn.close()
    except Exception as exc:
        raise SystemExit(f"[walk_forward] cannot reach Postgres: {exc}") from exc

    total_rows = 0
    total_anchors = 0
    cells_skipped = 0
    for lat, lon in cells:
        cell = (
            gold[(gold["lat"] == lat) & (gold["lon"] == lon)]
            .dropna(subset=feature_columns)
            .sort_values("observed_at")
            .reset_index(drop=True)
        )
        if len(cell) < spec.seq_in:
            cells_skipped += 1
            continue

        anchor_ts, preds = backtest_one_cell(
            cell_rows=cell,
            feature_columns=feature_columns,
            feature_mean=feature_mean,
            feature_std=feature_std,
            model=model,
            device=device,
            spec=spec,
            stride_hours=stride_hours,
            anchor_start=anchor_lo,
            anchor_end=t_max,
        )
        if len(anchor_ts) == 0:
            cells_skipped += 1
            continue

        loc_id = location_ids.get((lat, lon))
        if loc_id is None:
            print(
                f"[walk_forward] no dim_location for ({lat}, {lon}); skipping",
                file=sys.stderr,
            )
            cells_skipped += 1
            continue

        conn = connect_from_env()
        try:
            written = _persist_cell(
                conn=conn,
                model_version=walkforward_version,
                location_id=loc_id,
                anchor_ts=anchor_ts,
                preds=preds,
                seq_out=spec.seq_out,
                seq_in=spec.seq_in,
            )
        finally:
            conn.close()
        total_rows += written
        total_anchors += len(anchor_ts)

    print(
        f"[walk_forward] inserted={total_rows} rows  "
        f"anchors={total_anchors}  cells_skipped={cells_skipped}"
    )

    # Backfill actuals + refresh backtest_groups so the dashboard sees the new
    # rows in both surfaces (combobox via prediction_made_at, BT table via mse).
    conn = connect_from_env()
    try:
        with transaction(conn):
            filled = backfill_actuals_for_version(conn, walkforward_version)
        with transaction(conn):
            with conn.cursor() as cur:
                cur.execute(BACKTEST_GROUPS_UPSERT_SQL)
                grouped = cur.rowcount
    finally:
        conn.close()

    print(
        f"[walk_forward] done: model_version={walkforward_version} "
        f"actuals_filled={filled} backtest_groups_upserted={grouped}"
    )


if __name__ == "__main__":
    main()
