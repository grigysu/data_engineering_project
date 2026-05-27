# Weather Data Pipeline + ML Forecasting

End-to-end data engineering + ML system that pulls weather data for **Armenia** (10×10 lat/lon grid), runs it through a Bronze→Silver→Gold lake with PySpark, lands it in a Postgres star schema, trains a PyTorch LSTM, persists every prediction so it can be compared with actuals later, and is operated from a single Streamlit dashboard. Airflow drives the daily ingest + retrain.

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
                ┌────────────────────────────────────────────────────────┐
                │                    Airflow DAG (:8081)                 │
                │  ingest_archive → bronze → silver → gold → warehouse   │
                │                              ↘ train_model →           │
                │                              ↘ walk_forward →          │
                │                                backfill_actuals        │
                └────────────────────────────────────────────────────────┘
                                       │
   ┌───────────┐    ┌─────────────┐    │    ┌──────────────┐       ┌──────────────┐
   │ Open-Meteo│ →  │  Ingestion  │ →  │ →  │   PySpark    │ ----→ │  Postgres    │
   │ (archive) │    │ (additive)  │    │    │   ETL +      │       │  star schema │
   └───────────┘    └─────────────┘    │    │   features   │       │  + predictions│
                          │            │    └──────────────┘       │  + models    │
                          ▼            │           │               └──────────────┘
                  ┌─────────────────┐  │           │                    │   │
                  │  MinIO (:9000)  │  │           │                    │   │
                  │  bronze/silver/ │<─┘           │                    │   │
                  │  gold/manifest/ │              │                    │   │
                  └─────────────────┘              │                    │   │
                                                   ▼                    ▼   ▼
                          ┌────────────────────────────────────────────────────┐
                          │     Streamlit dashboard (:8501) — single control   │
                          │  status · data · train · predict-vs-actual · sql   │
                          └────────────────────────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                                  ┌──────────────┐
                                                                  │   Adminer    │
                                                                  │   (:8082)    │
                                                                  └──────────────┘
```

## Status

| Phase | Slice | Status |
|---|---|---|
| 0 | Scaffolding | ✅ |
| 1 | Ingestion → MinIO bronze + healthcheck preflight | ✅ |
| 2 (Spark) | Medallion ETL (bronze → silver → gold) + Hive metastore | ✅ |
| 3 | Postgres star schema | ✅ |
| 4 | PyTorch LSTM training + CSV logs + plots + GPU | ✅ |
| 5 | (retired) FastAPI inference — replaced by Streamlit in 2c | ✅→❌ |
| 6 | Airflow DAG `@daily` | ✅ |
| **2a** | MinIO-only storage; additive archive ingest via gold-coverage manifest | ✅ |
| **2b** | Prediction persistence + model registry + continue-training (`--resume`) | ✅ |
| **2c** | Streamlit unified control plane; FastAPI removed | ✅ |
| **2d** | Adminer + docs refresh + end-to-end verification | ✅ |
| **2e** | Walk-forward training: honest 7-day holdout + per-day backtest groups | ✅ |

## Quickstart (≈ 5 minutes once Docker is warm)

```powershell
# 1. Native venv (ingestion + ML training run here; dashboard runs in Docker)
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: GPU torch (Blackwell needs cu128+, see docs/WIKI.md#gpu)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Config
Copy-Item .env.example .env

# 3. Bring up the full stack
docker compose up -d

# 4. One-time DDL bootstrap (Hive externals + Postgres star schema + predictions/models)
docker cp warehouse/hive/create_external_tables.sql weather_spark_master:/tmp/
docker exec weather_spark_master /opt/spark/bin/spark-sql `
  --master spark://spark-master:7077 -f /tmp/create_external_tables.sql
Get-ChildItem warehouse/ddl/*.sql | Sort-Object Name | ForEach-Object {
    Get-Content $_.FullName -Raw | docker exec -i weather_postgres `
        psql -U weather -d weather_dw -v ON_ERROR_STOP=1
}

# 5. Open the dashboard — drive everything from there
start http://localhost:8501
```

## UIs (after `docker compose up -d`)

| URL | Service | Login |
|---|---|---|
| **http://localhost:8501** | **Streamlit dashboard — the main UI** | — |
| http://localhost:8082 | Adminer (Postgres SQL viewer) | server `postgres`, user `weather` |
| http://localhost:9001 | MinIO console | `minioadmin` / `minioadmin` |
| http://localhost:8080 | Spark master | — |
| http://localhost:8081 | Airflow web UI | `admin` / `admin` |

## Honest caveats

- **7–14 day weather forecasts from observations alone are hard.** Real operational forecasts use NWP (numerical weather prediction) simulators. An LSTM trained on a few months of observations will underperform Open-Meteo's own forecast at long horizons. The value here is the end-to-end *pipeline*, not the model.
- **Currently undertrained.** Open-Meteo's archive endpoint has been intermittently returning 504s; with limited bronze data the LSTM has learned the global mean and loses to a persistence baseline. Pipeline is correct, data volume is the bottleneck.
- **Windows + Spark natively is painful** (winutils, Hadoop config). Spark / Hive / Airflow / Postgres / MinIO / dashboard / Adminer all run in Docker; only the dev/ingestion/ML code runs in the native venv.

## Future enhancements (explicitly deferred)

Delta Lake / Iceberg, MLflow experiment tracking, Kafka streaming ingestion, distributed Spark cluster, cloud port (MinIO → S3, Postgres → Snowflake/BigQuery, Airflow → MWAA), Transformer model variant.
