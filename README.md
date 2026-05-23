# Weather Data Pipeline + ML Forecasting

End-to-end data engineering + ML system for weather forecasting over **Armenia** (10×10 lat/lon grid). Built as a portfolio-grade showcase of the modern data stack: object-store data lake, PySpark ETL, Hive-cataloged SQL access, dimensional warehouse, PyTorch sequence models, and Airflow orchestration.

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
                │                    Airflow DAG                      │
                │  ingest → bronze → silver → gold → train → deploy   │
                └─────────────────────────────────────────────────────┘
                                       │
   ┌───────────┐    ┌─────────────┐    │    ┌──────────────┐    ┌──────────────┐
   │ Open-Meteo│ →  │  Ingestion  │ →  │ →  │   PySpark    │ →  │  Postgres    │
   │  (free)   │    │  (Python)   │    │    │   ETL +      │    │  (Gold star  │
   └───────────┘    └─────────────┘    │    │   features   │    │   schema)    │
                          │            │    └──────────────┘    └──────────────┘
                          ▼            │           │                    │
                  ┌─────────────────┐  │           │                    ▼
                  │  MinIO (S3 API) │  │           │            ┌──────────────┐
                  │  bronze/        │  │           │            │   PyTorch    │
                  │  silver/        │◀─┘           │            │   training   │
                  │  gold/          │              │            │  (LSTM/Tx)   │
                  └─────────────────┘              │            └──────────────┘
                          │                        │                    │
                          ▼                        ▼                    ▼
                  ┌─────────────────────────────────────┐       ┌──────────────┐
                  │ Hive Metastore + Spark Thrift Server│       │  FastAPI     │
                  │  (SQL-on-Parquet, JDBC/BI access)   │       │  inference   │
                  └─────────────────────────────────────┘       └──────────────┘
```

## Stack

| Layer            | Tool                                                |
|------------------|-----------------------------------------------------|
| Data source      | Open-Meteo (free, no API key, hourly historical + forecast) |
| Ingestion        | Python + httpx + pydantic                           |
| Object store     | MinIO (S3-compatible, local)                        |
| Lake format      | Parquet, medallion layout (bronze / silver / gold)  |
| ETL              | PySpark (single-node, in Docker)                    |
| SQL catalog      | Hive Metastore + Spark Thrift Server                |
| Warehouse        | PostgreSQL — star schema                            |
| ML               | PyTorch (LSTM baseline, Transformer stretch)        |
| Inference        | FastAPI                                             |
| Orchestration    | Airflow (LocalExecutor, in Docker)                  |

## Repository Layout

```
data_engine/
├── docker-compose.yml          # Phase 1: minio + postgres. Later phases add spark/hive/airflow.
├── .env.example                # Copy to .env and adjust
├── requirements.txt            # Native venv dependencies
│
├── ingestion/                  # Python jobs that pull Open-Meteo data → bronze
│   ├── grid.py                 # Armenia 10x10 grid generation (implemented)
│   ├── openmeteo_client.py     # async HTTP client            (Phase 1)
│   ├── schemas.py              # pydantic models               (Phase 1)
│   └── run_ingest.py           # CLI entrypoint                (Phase 1)
│
├── spark_jobs/                 # PySpark ETL — runs inside the Spark container
│   ├── bronze_to_silver.py     # parse, dedupe, type-cast      (Phase 2)
│   ├── silver_to_gold.py       # feature engineering           (Phase 2)
│   ├── build_training_set.py   # window into ML sequences      (Phase 2)
│   └── load_to_warehouse.py    # Spark → Postgres via JDBC     (Phase 3)
│
├── warehouse/
│   ├── ddl/                    # Postgres star-schema DDL      (Phase 3)
│   └── hive/                   # External-table DDL for Spark SQL (Phase 2)
│
├── ml/
│   ├── dataset.py              # PyTorch Dataset over gold parquet (Phase 4)
│   ├── models/                 # lstm.py, transformer.py        (Phase 4)
│   ├── train.py                # train loop, checkpointing      (Phase 4)
│   └── evaluate.py             # backtest                       (Phase 4)
│
├── serving/
│   ├── api.py                  # FastAPI app                    (Phase 5)
│   └── predictor.py            # load checkpoint + featurize    (Phase 5)
│
├── airflow/
│   └── dags/weather_pipeline.py  # end-to-end DAG               (Phase 6)
│
├── docker/
│   └── postgres-init/          # SQL files auto-run on container boot
│
└── tests/
```

## Local Setup (Windows)

### 1. Python venv

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment

```powershell
Copy-Item .env.example .env
# Edit .env if you want non-default passwords / endpoints
```

### 3. Docker Desktop

Make sure Docker Desktop is **running** (whale icon steady in the system tray). Sign-in is not required. Then in a fresh PowerShell:

```powershell
docker compose up -d
```

This brings up MinIO (S3 API on `:9000`, console on `:9001`) and Postgres (`:5432`). The bootstrap container creates the `weather-lake` bucket with `bronze/`, `silver/`, `gold/` prefixes.

- MinIO console: http://localhost:9001 (login: `minioadmin` / `minioadmin`)
- Postgres: `psql -h localhost -U weather -d weather_dw`

## Region

Default region is **Armenia**, bounding box `(38.84°N, 43.45°E) → (41.30°N, 46.63°E)`, generated as a 10×10 grid = 100 points. Adjust via env vars in `.env`:

```
GRID_LAT_MIN=...
GRID_LAT_MAX=...
GRID_LON_MIN=...
GRID_LON_MAX=...
GRID_SIZE=10
```

Preview the grid:

```powershell
python -m ingestion.grid
```

## Build Phases

| Phase | Working slice                                          | Status |
|-------|--------------------------------------------------------|--------|
| 0     | Repo scaffolding (this commit)                         | ✅     |
| 1     | Ingestion writes bronze Parquet to MinIO               | ⏳     |
| 2     | Spark ETL + Hive-cataloged silver/gold tables          | ⏳     |
| 3     | Postgres star schema populated from gold               | ⏳     |
| 4     | PyTorch model trained, checkpoint on disk              | ⏳     |
| 5     | FastAPI `/forecast` endpoint live                      | ⏳     |
| 6     | Airflow DAG runs the whole pipeline end-to-end         | ⏳     |

See the [project plan](../../.claude/plans/this-is-my-project-snazzy-hammock.md) for details on each phase.

## Honest Caveats

- **7–14 day forecasts from observations alone are hard.** Operational weather forecasts use NWP (numerical weather prediction) models. This project's ML model will likely underperform Open-Meteo's own NWP-based forecast at long horizons — that's expected; the value is the end-to-end pipeline, not state-of-the-art meteorology.
- **Windows + Spark natively is painful** (winutils, Hadoop config). The Spark/Hive services run in Docker; only the dev/ingestion/ML code runs in the native venv.

## Future Enhancements (deferred)

Delta Lake / Iceberg, MLflow experiment tracking, Kafka streaming ingestion, distributed Spark cluster, cloud port (MinIO → S3, Postgres → Snowflake/BigQuery, Airflow → MWAA).
