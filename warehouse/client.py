"""Postgres client for the things outside the Spark full-refresh cycle.

`spark_jobs/load_to_warehouse.py` rebuilds the star schema each run (dims
via upsert, fact via TRUNCATE+INSERT). The `predictions` table is kept
outside that cycle so prediction history survives reloads — this client is
what writes to it. Model identity lives entirely in `checkpoints/best.pt`;
there is no Postgres registry.

All functions take a psycopg2 connection — callers manage the lifecycle.
That keeps the module easy to test (mock the connection) and avoids a
hidden global pool.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import psycopg2


@dataclass(frozen=True)
class PredictionRow:
    target_time: datetime
    predicted_value: float


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
    seq_in: int | None = None,
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
            seq_in,
        )
        for r in rows
    ]
    if not items:
        return 0
    sql = (
        "INSERT INTO predictions "
        "(model_version, location_id, prediction_made_at, target_time, predicted_value, seq_in) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
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


def backfill_actuals_for_version(conn, model_version: str) -> int:
    """Like `backfill_actuals` but scoped to a single model_version.

    Used by the walk-forward backtest (Phase 2e) to fill actuals only for its
    own freshly-inserted rows, without doing a project-wide UPDATE.
    """
    sql = """
        UPDATE predictions p
        SET actual_value = f.temperature_2m,
            actual_filled_at = NOW()
        FROM fact_weather_observations f
        JOIN dim_time t ON t.time_id = f.time_id
        WHERE p.model_version = %s
          AND p.actual_value IS NULL
          AND p.location_id = f.location_id
          AND p.target_time = t.observed_at
          AND f.dataset = 'archive'
          AND f.temperature_2m IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model_version,))
        return cur.rowcount


def list_location_ids(conn) -> dict[tuple[float, float], int]:
    """Snapshot dim_location into a {(lat, lon): location_id} lookup."""
    with conn.cursor() as cur:
        cur.execute("SELECT location_id, lat, lon FROM dim_location")
        return {
            (float(lat), float(lon)): int(loc_id) for loc_id, lat, lon in cur.fetchall()
        }
