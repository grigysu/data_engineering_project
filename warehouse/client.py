"""Postgres client for the things outside the Spark full-refresh cycle.

`spark_jobs/load_to_warehouse.py` TRUNCATEs dim/fact tables on every run.
Predictions + model registry must survive those rebuilds, so they go
through this client instead.

All functions take a psycopg2 connection — callers manage the lifecycle.
That keeps the module easy to test (mock the connection) and avoids a
hidden global pool.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import psycopg2


@dataclass(frozen=True)
class PredictionRow:
    target_time: datetime
    predicted_value: float


@dataclass(frozen=True)
class ModelInfo:
    model_version: str
    trained_at: datetime
    data_range_start: date | None
    data_range_end: date | None
    gold_row_count: int | None
    best_val_mse: float | None
    epochs: int | None
    is_best: bool
    checkpoint_path: str | None
    hyperparams: dict | None
    notes: str | None


def connect_from_env():
    """psycopg2 connection from POSTGRES_* env vars.

    Defaults match docker-compose; override `POSTGRES_HOST` to 'postgres'
    when running inside an Airflow container.
    """
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "weather_dw"),
        user=os.getenv("POSTGRES_USER", "weather"),
        password=os.getenv("POSTGRES_PASSWORD", "weather"),
    )


@contextmanager
def transaction(conn):
    """Commit on clean exit, rollback on exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------- predictions ----------


def insert_predictions(
    conn,
    *,
    model_version: str,
    location_id: int,
    prediction_made_at: datetime,
    rows: Iterable[PredictionRow],
) -> int:
    """Bulk-insert one /forecast call's worth of prediction rows.

    Idempotent via the UNIQUE (model_version, location_id, made_at, target_time)
    constraint: re-inserting the same call is a no-op.
    """
    items = [
        (
            model_version,
            location_id,
            prediction_made_at,
            r.target_time,
            r.predicted_value,
        )
        for r in rows
    ]
    if not items:
        return 0
    sql = (
        "INSERT INTO predictions "
        "(model_version, location_id, prediction_made_at, target_time, predicted_value) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (model_version, location_id, prediction_made_at, target_time) "
        "DO NOTHING"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, items)
        return cur.rowcount


def backfill_actuals(conn) -> int:
    """Fill predictions.actual_value from fact_weather_observations.

    Join on (location_id, target_time = dim_time.observed_at, dataset='archive')
    so we only resolve actuals from real observations, not forecasts.
    Returns the number of rows updated.
    """
    sql = """
        UPDATE predictions p
        SET actual_value = f.temperature_2m,
            actual_filled_at = NOW()
        FROM fact_weather_observations f
        JOIN dim_time t ON t.time_id = f.time_id
        WHERE p.actual_value IS NULL
          AND p.location_id = f.location_id
          AND p.target_time = t.observed_at
          AND f.dataset = 'archive'
          AND f.temperature_2m IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount


def list_location_ids(conn) -> dict[tuple[float, float], int]:
    """Snapshot dim_location into a {(lat, lon): location_id} lookup."""
    with conn.cursor() as cur:
        cur.execute("SELECT location_id, lat, lon FROM dim_location")
        return {
            (float(lat), float(lon)): int(loc_id) for loc_id, lat, lon in cur.fetchall()
        }


# ---------- model registry ----------


def register_model(
    conn,
    *,
    model_version: str,
    trained_at: datetime,
    data_range_start: date | None,
    data_range_end: date | None,
    gold_row_count: int | None,
    best_val_mse: float | None,
    epochs: int | None,
    checkpoint_path: str | None,
    hyperparams: dict | None,
    notes: str | None = None,
) -> None:
    """Upsert a row into `models`. Clears is_best on all rows then sets it on
    the new lowest-val-mse model.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO models (
                model_version, trained_at, data_range_start, data_range_end,
                gold_row_count, best_val_mse, epochs, checkpoint_path,
                hyperparams, notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (model_version) DO UPDATE SET
                trained_at = EXCLUDED.trained_at,
                data_range_start = EXCLUDED.data_range_start,
                data_range_end = EXCLUDED.data_range_end,
                gold_row_count = EXCLUDED.gold_row_count,
                best_val_mse = EXCLUDED.best_val_mse,
                epochs = EXCLUDED.epochs,
                checkpoint_path = EXCLUDED.checkpoint_path,
                hyperparams = EXCLUDED.hyperparams,
                notes = EXCLUDED.notes
            """,
            (
                model_version,
                trained_at,
                data_range_start,
                data_range_end,
                gold_row_count,
                best_val_mse,
                epochs,
                checkpoint_path,
                json.dumps(hyperparams) if hyperparams is not None else None,
                notes,
            ),
        )
        # Re-elect the best model (lowest non-null val MSE wins).
        cur.execute("UPDATE models SET is_best = FALSE WHERE is_best")
        cur.execute(
            """
            UPDATE models SET is_best = TRUE
            WHERE model_version = (
                SELECT model_version FROM models
                WHERE best_val_mse IS NOT NULL
                ORDER BY best_val_mse ASC, trained_at DESC
                LIMIT 1
            )
            """
        )


def get_best_model(conn) -> ModelInfo | None:
    """Return the row with is_best=TRUE, or None if the registry is empty."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_version, trained_at, data_range_start, data_range_end,
                   gold_row_count, best_val_mse, epochs, is_best,
                   checkpoint_path, hyperparams, notes
            FROM models
            WHERE is_best
            LIMIT 1
            """
        )
        row = cur.fetchone()
    if row is None:
        return None
    return ModelInfo(
        model_version=row[0],
        trained_at=row[1],
        data_range_start=row[2],
        data_range_end=row[3],
        gold_row_count=row[4],
        best_val_mse=row[5],
        epochs=row[6],
        is_best=bool(row[7]),
        checkpoint_path=row[8],
        hyperparams=row[9],
        notes=row[10],
    )


def list_models(conn, limit: int = 50) -> list[ModelInfo]:
    """Recent models, newest first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_version, trained_at, data_range_start, data_range_end,
                   gold_row_count, best_val_mse, epochs, is_best,
                   checkpoint_path, hyperparams, notes
            FROM models
            ORDER BY trained_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        ModelInfo(
            model_version=r[0],
            trained_at=r[1],
            data_range_start=r[2],
            data_range_end=r[3],
            gold_row_count=r[4],
            best_val_mse=r[5],
            epochs=r[6],
            is_best=bool(r[7]),
            checkpoint_path=r[8],
            hyperparams=r[9],
            notes=r[10],
        )
        for r in rows
    ]
