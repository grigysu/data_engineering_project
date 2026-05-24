"""End-to-end weather pipeline DAG.

Runs the full chain we've been triggering by hand:

    ingest_forecast → bronze_to_silver → silver_to_gold
                                            → load_warehouse
                                            → train_model

Ingestion + training run inside the Airflow container itself (which has
our project code mounted at /opt/weather and the runtime deps installed
via the custom image). Spark jobs run via `docker exec` into the
spark-master container — Airflow can do this because the host docker
socket is bind-mounted in.

Run/inspect from http://localhost:8081 (admin/admin).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/weather"

# S3/MinIO env vars exposed to the Python tasks so they can write to
# bronze and read gold straight from MinIO without a local mirror step.
S3_ENV = (
    "STORAGE_BACKEND=s3 "
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
    description="Ingest Open-Meteo → bronze/silver/gold on MinIO → Postgres warehouse → LSTM training.",
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["weather", "end-to-end"],
) as dag:
    # Phase 1: pull the latest 7-day forecast for every grid point.
    # --skip-check because the preflight healthcheck assumes we always need
    # archive too; here we only use forecast.
    ingest_forecast = BashOperator(
        task_id="ingest_forecast",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"{S3_ENV} python -m ingestion.run_ingest "
            "--forecast --forecast-days 7 --concurrency 5 --skip-check"
        ),
    )

    # Phase 2a: JSON → Parquet, explode hourly arrays, partition by (dataset, date).
    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/bronze_to_silver.py",
    )

    # Phase 2a: time + lag + rolling features.
    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/silver_to_gold.py",
    )

    # Phase 3: TRUNCATE + INSERT into fact + dims via JDBC.
    load_warehouse = BashOperator(
        task_id="load_warehouse",
        bash_command=f"{SPARK_SUBMIT} /opt/jobs/load_to_warehouse.py",
    )

    # Phase 4: retrain the LSTM against the freshly-rebuilt gold table.
    # Reads gold straight from MinIO via the s3:// path support added to
    # ml.dataset.load_gold; checkpoints land in /opt/weather/checkpoints.
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

    ingest_forecast >> bronze_to_silver >> silver_to_gold
    silver_to_gold >> [load_warehouse, train_model]
