"""End-to-end weather pipeline DAG.

Runs the full chain:

    ingest_archive → bronze_to_silver → silver_to_gold
                                          → load_warehouse
                                          → train_model

Daily run: additively ingests the last 30 days of *archive* (real observations)
from Open-Meteo, then re-builds silver/gold/warehouse and retrains.

Manual trigger: pass `conf={"start_date": "...", "end_date": "...", "force": false}`
to backfill a custom range. The Streamlit dashboard (Phase 2c) calls this DAG
via the Airflow REST API with custom conf.

Phase 2a: switched from `--forecast` (Open-Meteo's own model output) to
`--backfill` against the archive endpoint — bronze now means real observations.
Ingest is additive (gold-coverage manifest), so daily runs are cheap once the
backfill has caught up.

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

    # TRUNCATE + INSERT into fact + dims via JDBC.
    load_warehouse = BashOperator(
        task_id="load_warehouse",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/load_to_warehouse.py",
    )

    # Retrain the LSTM against the freshly-rebuilt gold table.
    # Reads gold straight from MinIO via the s3:// path support in ml.dataset.
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"{S3_ENV} python -m ml.train "
            "--gold s3://weather-lake/gold/weather_features "
            "--epochs 20 --batch-size 16 "
            "--checkpoint checkpoints/best.pt"
        ),
    )

    ingest_archive >> bronze_to_silver >> silver_to_gold
    silver_to_gold >> [load_warehouse, train_model]
