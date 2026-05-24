"""Generate one fresh forecast per grid cell and persist to `predictions`.

Runs as an Airflow task after `train_model` + `load_warehouse`. Reuses
`serving.predictor.Predictor` for checkpoint+gold loading and grid
enumeration; reuses `build_inference_batch` + `_to_utc_dt` from
`ml.backtest`. Each cell contributes one anchor (the most recent valid
hour) × `seq_out` rows per scheduler invocation.

Usage:
    python -m ml.predict_all --checkpoint checkpoints/best.pt \\
        --gold s3://weather-lake/gold/weather_features
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from ml.backtest import _to_utc_dt, build_inference_batch
from serving.predictor import Predictor
from warehouse.client import (
    PredictionRow,
    connect_from_env,
    insert_predictions,
    list_location_ids,
    transaction,
)


def predict_one_cell(
    predictor: Predictor, lat: float, lon: float
) -> list[PredictionRow] | None:
    """Forecast `seq_out` hours from the most recent valid anchor for (lat, lon).

    Returns the row list, or None when the cell has fewer than `seq_in`
    rows with non-NaN features (so we can't form one inference window).
    """
    rows = (
        predictor.gold[(predictor.gold["lat"] == lat) & (predictor.gold["lon"] == lon)]
        .dropna(subset=list(predictor.feature_columns))
        .sort_values("observed_at")
    )
    if len(rows) < predictor.spec.seq_in:
        return None

    feats = rows[list(predictor.feature_columns)].to_numpy(dtype=np.float32)
    feats_norm = (feats - predictor.feature_mean) / predictor.feature_std
    batch = build_inference_batch(feats_norm, [len(rows) - 1], predictor.spec.seq_in)
    with torch.no_grad():
        x_t = torch.from_numpy(batch).to(predictor.device)
        pred = predictor.model(x_t).cpu().numpy()[0]

    anchor = _to_utc_dt(rows["observed_at"].iloc[-1])
    return [
        PredictionRow(
            target_time=anchor + timedelta(hours=k + 1),
            predicted_value=float(pred[k]),
        )
        for k in range(predictor.spec.seq_out)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-cell forecast persistence (auto-pipeline)."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gold", required=True, help="s3:// URI of the gold parquet.")
    args = parser.parse_args()

    # Predictor handles checkpoint + gold loading. persist=False because we
    # batch the inserts ourselves below (per cell, in one transaction).
    predictor = Predictor(args.checkpoint, args.gold, persist=False)

    try:
        conn = connect_from_env()
        try:
            loc_ids = list_location_ids(conn)
        finally:
            conn.close()
    except Exception as exc:
        raise SystemExit(f"[predict_all] cannot reach Postgres: {exc}") from exc

    made_at = datetime.now(timezone.utc)
    if predictor.model_version is None:
        raise SystemExit(
            "[predict_all] checkpoint has no model_version; refusing to write "
            "predictions without a traceable identity."
        )

    n_written = n_skipped_history = n_skipped_dim = 0
    for lat, lon in predictor.grid_points():
        rows = predict_one_cell(predictor, lat, lon)
        if rows is None:
            n_skipped_history += 1
            continue
        loc_id = loc_ids.get((lat, lon))
        if loc_id is None:
            print(
                f"[predict_all] no dim_location row for ({lat}, {lon}); skipping",
                file=sys.stderr,
            )
            n_skipped_dim += 1
            continue
        conn = connect_from_env()
        try:
            with transaction(conn):
                n_written += insert_predictions(
                    conn,
                    model_version=predictor.model_version,
                    location_id=loc_id,
                    prediction_made_at=made_at,
                    rows=rows,
                )
        finally:
            conn.close()

    print(
        f"[predict_all] done: model_version={predictor.model_version} "
        f"rows_inserted={n_written} "
        f"cells_skipped_history={n_skipped_history} "
        f"cells_skipped_dim={n_skipped_dim}"
    )


if __name__ == "__main__":
    main()
