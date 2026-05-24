"""Unit tests for the new stage-builders in spark_jobs/load_to_warehouse.py.

The big change in this restructure: dims no longer carry a Spark-computed
surrogate key column — Postgres assigns IDs via IDENTITY, the loader
upserts on the natural key. These tests pin the new column shapes so
neither builder accidentally re-introduces a `_id` field.

Skipped when pyspark isn't installed locally (production Spark only runs
in Docker; the venv keeps it optional).
"""

from __future__ import annotations

import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

from spark_jobs.load_to_warehouse import (  # noqa: E402
    build_dim_location_stage,
    build_dim_time_stage,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    s = (
        SparkSession.builder.master("local[1]")
        .appName("test_load_to_warehouse_upsert")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield s
    s.stop()


SILVER_SCHEMA_COLS = [
    "region",
    "lat",
    "lon",
    "elevation",
    "observed_at",
    "dataset",
    "temperature_2m",
]


def _silver_rows(spark: SparkSession):
    """Two cells × two timestamps × archive dataset = 4 rows."""
    from datetime import datetime

    rows = [
        ("armenia", 40.0, 44.0, 1000.0, datetime(2026, 5, 24, 12), "archive", 18.5),
        ("armenia", 40.0, 44.0, 1000.0, datetime(2026, 5, 24, 13), "archive", 19.0),
        ("armenia", 41.0, 45.0, 1500.0, datetime(2026, 5, 24, 12), "archive", 12.0),
        ("armenia", 41.0, 45.0, 1500.0, datetime(2026, 5, 24, 13), "archive", 12.5),
    ]
    return spark.createDataFrame(rows, SILVER_SCHEMA_COLS)


def test_dim_location_stage_columns(spark: SparkSession):
    silver = _silver_rows(spark)
    dim = build_dim_location_stage(silver)
    assert set(dim.columns) == {"region", "lat", "lon", "elevation"}
    # No surrogate key column — Postgres IDENTITY owns it.
    assert "location_id" not in dim.columns


def test_dim_location_stage_distinct(spark: SparkSession):
    silver = _silver_rows(spark)
    dim = build_dim_location_stage(silver)
    # 4 silver rows → 2 distinct cells.
    assert dim.count() == 2


def test_dim_time_stage_columns(spark: SparkSession):
    silver = _silver_rows(spark)
    dim = build_dim_time_stage(silver)
    expected = {
        "observed_at",
        "date",
        "hour",
        "day_of_week",
        "month",
        "year",
        "season",
        "is_weekend",
    }
    assert set(dim.columns) == expected
    assert "time_id" not in dim.columns


def test_dim_time_stage_distinct(spark: SparkSession):
    silver = _silver_rows(spark)
    dim = build_dim_time_stage(silver)
    # 4 silver rows → 2 distinct timestamps.
    assert dim.count() == 2


def test_dim_time_stage_season_attribution(spark: SparkSession):
    silver = _silver_rows(spark)
    dim = build_dim_time_stage(silver)
    # May 24 falls in 'spring' per _season_expr month-grouping.
    seasons = {r["season"] for r in dim.collect()}
    assert seasons == {"spring"}
