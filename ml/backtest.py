"""Walk-forward backtest over a trained checkpoint.

For each grid cell, slides a `seq_in`-hour window through historical gold
rows and records the model's `seq_out`-hour forecast at every anchor. Each
row is written to the existing `predictions` table under a unique
`model_version = "backtest:<orig>:<run_ts>"`, then `actual_value` is filled
in-place from `fact_weather_observations` via the same join the nightly
Airflow task uses (scoped to this run only).

The same loop naturally produces the right-edge forecast (anchor = most
recent observation) whose targets are in the future — those rows keep
`actual_value = NULL` and render as the "what does the model say next?"
tail on the dashboard.

Usage:
    python -m ml.backtest --checkpoint checkpoints/best.pt \\
        --gold s3://weather-lake/gold/weather_features \\
        --start 2026-04-01 --end 2026-05-24 --stride-hours 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.dataset import WindowSpec, load_gold
from ml.models.lstm import WeatherLSTM
from warehouse.client import (
    PredictionRow,
    backfill_actuals_for_version,
    connect_from_env,
    insert_predictions,
    list_location_ids,
    transaction,
)


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
) -> int:
    """Insert one batch of backtest rows. Returns total rows written."""
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
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of a trained LSTM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gold", required=True, help="s3:// URI of the gold parquet.")
    parser.add_argument(
        "--start",
        help="ISO date — first anchor timestamp to include (UTC). Default: no lower bound.",
    )
    parser.add_argument(
        "--end",
        help="ISO date — last anchor timestamp to include (UTC). Default: no upper bound.",
    )
    parser.add_argument(
        "--stride-hours",
        type=int,
        default=1,
        help="Hours between consecutive anchors. 1 = one prediction per hour.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        help="Restrict to a single grid cell. Requires --lon.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Restrict to a single grid cell. Requires --lat.",
    )
    args = parser.parse_args()

    if (args.lat is None) != (args.lon is None):
        raise SystemExit("[backtest] --lat and --lon must be given together.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backtest] device={device}  checkpoint={args.checkpoint}")

    model, feature_columns, spec, feature_mean, feature_std, orig_version = (
        _load_checkpoint(args.checkpoint, device)
    )
    print(
        f"[backtest] checkpoint loaded: orig_version={orig_version} "
        f"seq_in={spec.seq_in} seq_out={spec.seq_out}"
    )

    print(f"[backtest] loading gold from {args.gold}")
    gold = load_gold(args.gold)
    if args.lat is not None:
        gold = gold[(gold["lat"] == args.lat) & (gold["lon"] == args.lon)]

    cells = sorted(
        {(float(lat), float(lon)) for lat, lon in gold[["lat", "lon"]].to_numpy()}
    )
    if not cells:
        raise SystemExit("[backtest] no rows match the requested cell/range.")

    try:
        conn = connect_from_env()
        try:
            location_ids = list_location_ids(conn)
        finally:
            conn.close()
    except Exception as exc:
        raise SystemExit(f"[backtest] cannot reach Postgres: {exc}") from exc

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    backtest_version = f"backtest:{orig_version}:{run_ts}"
    anchor_start = pd.Timestamp(args.start) if args.start else None
    anchor_end = pd.Timestamp(args.end) if args.end else None

    total_rows = 0
    for lat, lon in cells:
        cell = (
            gold[(gold["lat"] == lat) & (gold["lon"] == lon)]
            .dropna(subset=feature_columns)
            .sort_values("observed_at")
            .reset_index(drop=True)
        )
        if len(cell) < spec.seq_in:
            print(
                f"[backtest] ({lat}, {lon}): only {len(cell)} usable rows "
                f"(need {spec.seq_in}); skipping",
                file=sys.stderr,
            )
            continue

        anchor_ts, preds = backtest_one_cell(
            cell_rows=cell,
            feature_columns=feature_columns,
            feature_mean=feature_mean,
            feature_std=feature_std,
            model=model,
            device=device,
            spec=spec,
            stride_hours=args.stride_hours,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
        )
        if len(anchor_ts) == 0:
            print(f"[backtest] ({lat}, {lon}): no anchors in range; skipping")
            continue

        loc_id = location_ids.get((lat, lon))
        if loc_id is None:
            print(
                f"[backtest] no dim_location row for ({lat}, {lon}); skipping persist",
                file=sys.stderr,
            )
            continue

        conn = connect_from_env()
        try:
            written = _persist_cell(
                conn=conn,
                model_version=backtest_version,
                location_id=loc_id,
                anchor_ts=anchor_ts,
                preds=preds,
                seq_out=spec.seq_out,
            )
        finally:
            conn.close()
        total_rows += written
        print(f"[backtest] ({lat}, {lon}): {len(anchor_ts)} anchors → {written} rows")

    conn = connect_from_env()
    try:
        with transaction(conn):
            filled = backfill_actuals_for_version(conn, backtest_version)
    finally:
        conn.close()

    print(
        f"[backtest] done: model_version={backtest_version} "
        f"inserted={total_rows} actuals_filled={filled}"
    )


if __name__ == "__main__":
    main()
