# Weather Data Pipeline + ML Forecasting

End-to-end data engineering + ML system that pulls weather data for **Armenia** (10×10 lat/lon grid), runs it through a Bronze→Silver→Gold lake with PySpark, lands it in a Postgres star schema, trains a PyTorch LSTM, serves predictions over HTTP, and orchestrates the whole thing with Airflow.

Built as a portfolio piece showing the full modern data stack glued together on one laptop.

---

## Documentation map

| Doc | Read when… |
|---|---|
| **[docs/PROGRESS.md](docs/PROGRESS.md)** | You're an ML engineer who wants a guided tour of what each layer is and *why* it exists. |
| **[docs/USAGE.md](docs/USAGE.md)** | You want to actually run things — every command, every CLI flag, common workflows, troubleshooting. |
| **[docs/WIKI.md](docs/WIKI.md)** | You're looking something up — env vars, ports, schemas, services, file layout. |

If you read only one section of any doc, read the [big picture](docs/PROGRESS.md#the-big-picture) in PROGRESS.

---

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
                │                    Airflow DAG (:8081)              │
                │  ingest → bronze → silver → gold → warehouse → train│
                └─────────────────────────────────────────────────────┘
                                       │
   ┌───────────┐    ┌─────────────┐    │    ┌──────────────┐    ┌──────────────┐
   │ Open-Meteo│ →  │  Ingestion  │ →  │ →  │   PySpark    │ →  │  Postgres    │
   │  (free)   │    │  (Python)   │    │    │   ETL +      │    │  star schema │
   └───────────┘    └─────────────┘    │    │   features   │    └──────────────┘
                          │            │    └──────────────┘            │
                          ▼            │           │                    ▼
                  ┌─────────────────┐  │           │           ┌──────────────┐
                  │  MinIO (:9000)  │  │           │           │ PyTorch LSTM │
                  │  bronze/silver/ │◀─┘           │           │   training   │
                  │  gold/          │              │           └──────────────┘
                  └─────────────────┘              │                    │
                          │                        ▼                    ▼
                          ▼               ┌────────────────────┐  ┌──────────────┐
                  ┌────────────────────┐  │ Hive Metastore     │  │  FastAPI     │
                  │ Spark SQL via      │──┤  (:9083) catalog   │  │  /forecast   │
                  │ external tables    │  └────────────────────┘  │   (:8000)    │
                  └────────────────────┘                          └──────────────┘
```

## Status

All 7 build phases complete and verified end-to-end:

| Phase | Slice | Status |
|---|---|---|
| 0 | Scaffolding | ✅ |
| 1 | Ingestion → MinIO bronze + healthcheck preflight | ✅ |
| 2a | Spark medallion ETL (bronze → silver → gold) | ✅ |
| 2b | Hive Metastore + SQL access via `spark-sql` | ✅ |
| 3 | Postgres star schema (`fact_weather_observations`, `dim_location`, `dim_time`) | ✅ |
| 4 | PyTorch LSTM training + CSV logs + matplotlib plots + GPU (CUDA) | ✅ |
| 5 | FastAPI inference (`/health`, `/grid_points`, `/forecast`) | ✅ |
| 6 | Airflow DAG runs the whole pipeline (`@daily`, manual-triggerable) | ✅ |

**Tests:** 26 passing (ingestion, healthcheck, dataset windowing, predictor, API).
**Lint/format:** ruff clean across 28 files.

## Quickstart (≈ 5 minutes once Docker is warm)

```powershell
# 1. Native venv (Phase 1 ingestion + Phase 4/5 ML+serving run here)
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: GPU torch (Blackwell needs cu128+, see docs/WIKI.md#gpu)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Config
Copy-Item .env.example .env

# 3. Bring up the full stack (~3 GB of images on first pull)
docker compose up -d

# 4. One-time: register Hive external tables over the existing parquet
docker cp warehouse/hive/create_external_tables.sql weather_spark_master:/tmp/
docker exec weather_spark_master /opt/spark/bin/spark-sql `
  --master spark://spark-master:7077 -f /tmp/create_external_tables.sql

# 5. Pull data, transform, train, serve, orchestrate — see docs/USAGE.md
python -m ingestion.run_ingest --forecast --forecast-days 7
docker exec weather_spark_master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 /opt/jobs/bronze_to_silver.py
# ... continues in docs/USAGE.md
```

## UIs (after `docker compose up -d`)

| URL | Service | Login |
|---|---|---|
| http://localhost:9001 | MinIO console | `minioadmin` / `minioadmin` |
| http://localhost:8080 | Spark master | — |
| http://localhost:8081 | Airflow web UI | `admin` / `admin` |
| http://localhost:8000/docs | FastAPI (when running) | — |

## Honest caveats

- **7–14 day weather forecasts from observations alone are hard.** Real operational forecasts use NWP (numerical weather prediction) simulators. An LSTM trained on a few months of observations will underperform Open-Meteo's own forecast at long horizons. The value here is the end-to-end *pipeline*, not the model.
- **Currently undertrained.** Open-Meteo's archive endpoint has been returning persistent 504s; we only have ~3 days of forecast data in bronze (≈ 288 silver rows, 60 training windows). The LSTM has learned the global mean and loses to a persistence baseline. Pipeline is correct, data volume is the bottleneck. See `predictions_vs_actual.png` for the visible failure mode.
- **Windows + Spark natively is painful** (winutils, Hadoop config). Spark / Hive / Airflow / Postgres / MinIO all run in Docker; only the dev/ingestion/ML code runs in the native venv.

## Future enhancements (explicitly deferred)

Delta Lake / Iceberg, MLflow experiment tracking, Kafka streaming ingestion, distributed Spark cluster, cloud port (MinIO → S3, Postgres → Snowflake/BigQuery, Airflow → MWAA), Transformer model variant.
