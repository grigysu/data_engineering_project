"""End-to-end weather pipeline DAG.

Runs the full chain (diamond joining at `walk_forward`):

    ingest_archive → bronze_to_silver → silver_to_gold ┬→ load_warehouse ─┐
                                                        └→ train_model ────┴→ walk_forward → backfill_actuals

Daily run: additively ingests the last 30 days of *archive* (real observations)
from Open-Meteo, rebuilds silver/gold/warehouse (upserting dims), re-trains
the LSTM on gold[<= T - cutoff_days] (honest holdout), then in a single
pass evaluates that model against `lookback_days` anchors AND produces the
operational forecast (rightmost anchor = T, targets in the future). The
user never has to click anything — the dashboard is read-only.

Manual trigger: pass `conf={"start_date": "...", "end_date": "...", "force": false}`
to backfill a custom range.

Run/inspect from http://localhost:8081 (admin/admin).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/weather"

# MinIO + S3 client env vars exposed to the Python tasks so they can write to
# bronze and read gold straight from MinIO without a local mirror step. The
# STORAGE_BACKEND var was dropped in Phase 2a (MinIO is the only backend).
S3_ENV = (
    "MINIO_BUCKET=weather-lake "
    "MINIO_ENDPOINT=http://minio:9000 "
    "MINIO_ROOT_USER=minioadmin "
    "MINIO_ROOT_PASSWORD=minioadmin "
    "S3_ENDPOINT=http://minio:9000 "
    "S3_ACCESS_KEY=minioadmin "
    "S3_SECRET_KEY=minioadmin "
    "REGION_NAME=armenia "
    "GRID_LAT_MIN=38.84 GRID_LAT_MAX=41.30 "
    "GRID_LON_MIN=43.45 GRID_LON_MAX=46.63 GRID_SIZE=10"
)

# Postgres env for the Python tasks that touch the warehouse directly
# (warehouse.actuals_backfill, ml.walk_forward).
WAREHOUSE_ENV = (
    "POSTGRES_HOST=postgres "
    "POSTGRES_PORT=5432 "
    "POSTGRES_DB=weather_dw "
    "POSTGRES_USER=weather "
    "POSTGRES_PASSWORD=weather"
)

# Spark submit invoked via the host docker daemon (socket is mounted in).
SPARK_SUBMIT = (
    "docker exec weather_spark_master "
    "/opt/spark/bin/spark-submit --master spark://spark-master:7077"
)


default_args = {
    "owner": "data_engine",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}


with DAG(
    dag_id="weather_pipeline",
    description=(
        "Ingest Open-Meteo archive (additive) → bronze/silver/gold on MinIO → "
        "Postgres warehouse → LSTM training."
    ),
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    params={
        # Default daily window: archive lags ~2 days, so pull yesterday-2d back
        # 30 days. Jinja-rendered at run time from the DagRun's logical date.
        "start_date": Param(
            "",
            type="string",
            description=(
                "Backfill start date YYYY-MM-DD. Empty = (logical_date - 32d)."
            ),
        ),
        "end_date": Param(
            "",
            type="string",
            description=("Backfill end date YYYY-MM-DD. Empty = (logical_date - 2d)."),
        ),
        "force": Param(
            False,
            type="boolean",
            description="Re-fetch every chunk even if the manifest says it's covered.",
        ),
    },
    tags=["weather", "end-to-end"],
) as dag:
    # Additive ingest of the archive endpoint. Empty start/end → defaults
    # computed at render time from `ds` (DAG logical date).
    ingest_archive = BashOperator(
        task_id="ingest_archive",
        bash_command=(
            'START="{{ params.start_date or macros.ds_add(ds, -32) }}"; '
            'END="{{ params.end_date or macros.ds_add(ds, -2) }}"; '
            'FORCE_FLAG="{{ "--force" if params.force else "" }}"; '
            f"cd {PROJECT_DIR} && "
            f"{S3_ENV} python -m ingestion.run_ingest "
            '--backfill "$START:$END" $FORCE_FLAG '
            "--concurrency 5 --skip-check"
        ),
    )

    # JSON → Parquet, explode hourly arrays, partition by (dataset, date).
    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/bronze_to_silver.py",
    )

    # Time + lag + rolling features.
    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/silver_to_gold.py",
    )

    # Upsert dims + TRUNCATE+INSERT fact via JDBC.
    load_warehouse = BashOperator(
        task_id="load_warehouse",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/load_to_warehouse.py",
    )

    # Retrain the LSTM against the freshly-rebuilt gold table. All hyperparams
    # live in config/train.yaml — pass nothing here so the YAML is the single
    # source of truth. `cutoff_days` reserves the most recent days as honest
    # holdout that walk_forward will score against.
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && {S3_ENV} {WAREHOUSE_ENV} python -m ml.train"
        ),
    )

    # Walk-forward: in one pass evaluates the freshly-trained model against
    # the last `lookback_days` of anchors AND produces the operational
    # forecast (rightmost anchor = T, targets in the future). Needs both a
    # trained checkpoint AND dim_location populated in Postgres.
    # Bumped execution_timeout: this does inference across ~lookback_days
    # anchors × ~100 cells; well under 30min but the default is tight.
    walk_forward = BashOperator(
        task_id="walk_forward",
        bash_command=(
            f"cd {PROJECT_DIR} && {S3_ENV} {WAREHOUSE_ENV} python -m ml.walk_forward"
        ),
        execution_timeout=timedelta(hours=1),
    )

    # Fill predictions.actual_value where the corresponding observation
    # just landed in fact_weather_observations (runs last so it sees the
    # freshly inserted forecasts too). walk_forward itself only backfills
    # its own walkforward:* rows; this catches anything else (e.g. legacy
    # rows whose targets just landed).
    backfill_actuals = BashOperator(
        task_id="backfill_actuals",
        bash_command=(
            f"cd {PROJECT_DIR} && {WAREHOUSE_ENV} python -m warehouse.actuals_backfill"
        ),
    )

    ingest_archive >> bronze_to_silver >> silver_to_gold
    silver_to_gold >> load_warehouse
    silver_to_gold >> train_model
    [load_warehouse, train_model] >> walk_forward >> backfill_actuals
