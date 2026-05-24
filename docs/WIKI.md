# Wiki — topical reference

Look things up here. Not a tutorial — for that, see [USAGE.md](USAGE.md); for a guided walkthrough, see [PROGRESS.md](PROGRESS.md).

## Table of contents

- [Service inventory](#service-inventory)
- [Ports](#ports)
- [Environment variables](#environment-variables)
- [Postgres databases](#postgres-databases)
- [MinIO bucket layout](#minio-bucket-layout)
- [File layout](#file-layout)
- [Data schemas](#data-schemas)
  - [Bronze JSON](#bronze-json)
  - [Silver / Gold Parquet](#silver--gold-parquet)
  - [Postgres star schema](#postgres-star-schema)
- [Dashboard reference](#dashboard-reference)
- [Airflow DAG reference](#airflow-dag-reference)
- [GPU](#gpu)
- [Git log + commit conventions](#git-log--commit-conventions)
- [External resources](#external-resources)

---

## Service inventory

| Container | Image | Purpose | Healthcheck |
|---|---|---|---|
| `weather_minio` | `minio/minio:latest` | S3-compatible object store | `/minio/health/live` |
| `weather_minio_init` | `minio/mc:latest` | One-shot: create `weather-lake/{bronze,silver,gold}` prefixes | — |
| `weather_postgres` | `postgres:16` | Backing DB for warehouse + Hive metastore + Airflow | `pg_isready` |
| `weather_spark_master` | `apache/spark:3.5.6` | Spark master + spark-submit launcher | (no explicit) |
| `weather_spark_worker` | `apache/spark:3.5.6` | Spark worker (2 cores, 2 GB) | (no explicit) |
| `weather_spark_init` | `apache/spark:3.5.6` | One-shot: `chown spark:spark /home/spark/.ivy2` | — |
| `weather_hive_metastore` | `apache/hive:4.0.0` | Standalone Hive Metastore, Thrift on :9083 | (no explicit) |
| `weather_hive_init` | `apache/hive:4.0.0` | One-shot: `schematool -initOrUpgradeSchema` | — |
| `weather_airflow_init` | `weather_airflow:latest` (custom) | One-shot: `airflow db migrate` + create admin user | — |
| `weather_airflow_webserver` | `weather_airflow:latest` | Airflow UI | — |
| `weather_airflow_scheduler` | `weather_airflow:latest` | Airflow scheduler (LocalExecutor) | — |
| `weather_dashboard` | `weather_dashboard:latest` (custom) | Streamlit control plane on :8501 | — |
| `weather_adminer` | `adminer:latest` | Postgres web SQL viewer on :8082 | — |

The custom `weather_airflow:latest` image is built from [docker/airflow/Dockerfile](../docker/airflow/Dockerfile): extends `apache/airflow:2.10.5-python3.11` with `docker.io` (CLI) + our runtime deps (httpx, pydantic, dotenv, boto3, pyarrow, pandas, psycopg2-binary, torch CPU, numpy, scikit-learn, matplotlib).

The custom `weather_dashboard:latest` image is built from [docker/dashboard/Dockerfile](../docker/dashboard/Dockerfile): slim `python:3.11-slim` + CPU torch + streamlit + plotly + our deps. Source is bind-mounted at runtime so dashboard edits don't require a rebuild.

## Ports

| Port (host) | Container | Purpose |
|---|---|---|
| **9000** | minio | S3 API |
| **9001** | minio | MinIO web console (`minioadmin`/`minioadmin`) |
| **5432** | postgres | Postgres (3 DBs — see below) |
| **8080** | spark-master | Spark master UI |
| **7077** | spark-master | Spark master RPC (`spark://spark-master:7077` from within docker net) |
| **4040** | spark-master | Spark application UI (only while a job runs) |
| **9083** | hive-metastore | Hive Thrift metastore |
| **8081** | airflow-webserver | Airflow web UI (`admin`/`admin`) |
| **8501** | dashboard | Streamlit control plane — main UI |
| **8082** | adminer | Postgres web SQL viewer |

## Environment variables

Loaded from `.env` (copy from `.env.example`). All have working defaults.

### Storage / MinIO

Phase 2a removed the local-FS backend. MinIO is the only storage; the `STORAGE_BACKEND` env var is gone.

| Var | Default | Used by |
|---|---|---|
| `MINIO_ROOT_USER` | `minioadmin` | MinIO + ingestion S3 client |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO + ingestion S3 client |
| `MINIO_ENDPOINT` | `http://localhost:9000` | Ingestion S3 client |
| `MINIO_BUCKET` | `weather-lake` | Ingestion + bootstrap |
| `MINIO_REGION` | `us-east-1` | Ingestion S3 client |

For PyArrow's S3FileSystem (used by `ml.dataset.load_gold` for `s3://` paths) and `ingestion.coverage.s3_client_from_env`:

| Var | Default |
|---|---|
| `S3_ENDPOINT` | `http://minio:9000` |
| `S3_ACCESS_KEY` | `minioadmin` |
| `S3_SECRET_KEY` | `minioadmin` |

### Postgres

| Var | Default | Used by |
|---|---|---|
| `POSTGRES_USER` | `weather` | Postgres super, warehouse owner |
| `POSTGRES_PASSWORD` | `weather` | — |
| `POSTGRES_DB` | `weather_dw` | Default DB |
| `POSTGRES_HOST` | `localhost` | Clients on host |
| `POSTGRES_PORT` | `5432` | — |

### Open-Meteo

| Var | Default |
|---|---|
| `OPENMETEO_BASE_URL` | `https://archive-api.open-meteo.com/v1/archive` |
| `OPENMETEO_FORECAST_URL` | `https://api.open-meteo.com/v1/forecast` |

### Region / grid

| Var | Default (Armenia) |
|---|---|
| `REGION_NAME` | `armenia` |
| `GRID_LAT_MIN` | `38.84` |
| `GRID_LAT_MAX` | `41.30` |
| `GRID_LON_MIN` | `43.45` |
| `GRID_LON_MAX` | `46.63` |
| `GRID_SIZE` | `10` (→ 10×10 = 100 points) |

### Ingestion / predictor / dashboard

| Var | Default | Used by |
|---|---|---|
| `INGEST_CONCURRENCY` | `5` | `ingestion.run_ingest` |
| `MODEL_CHECKPOINT` | `checkpoints/best.pt` | predictor + dashboard |
| `GOLD_PATH` | `s3://weather-lake/gold/weather_features` | predictor + dashboard |
| `CHECKPOINT_DIR` | `checkpoints` | dashboard (loss-curve discovery) |
| `WEATHER_DAG_ID` | `weather_pipeline` | dashboard |
| `AIRFLOW_BASE_URL` | `http://localhost:8081` | dashboard → Airflow REST API |
| `AIRFLOW_USER` | `admin` | dashboard |
| `AIRFLOW_PASSWORD` | `admin` | dashboard |
| `ADMINER_URL` | `http://localhost:8082` | dashboard (Browse page link) |

## Postgres databases

All three live in the single `weather_postgres` container.

| DB | Owner | Purpose | Initialized by |
|---|---|---|---|
| `weather_dw` | `weather` | Star schema (`fact_weather_observations`, `dim_location`, `dim_time`) + Phase 2b tables (`predictions`, `models`) | DDL in [warehouse/ddl/](../warehouse/ddl/), run by hand |
| `metastore_db` | `hive` | Hive Metastore's metadata (table defs, partitions) | `hive_init` runs `schematool -initOrUpgradeSchema` |
| `airflow_db` | `airflow` | Airflow metadata (DAG runs, task instances, users) | `airflow_init` runs `airflow db migrate` |

Init SQL for the `hive` and `airflow` users is in [docker/postgres-init/](../docker/postgres-init/). Runs **only on first postgres volume init**; for existing volumes, create manually:

```sql
CREATE USER hive WITH PASSWORD 'hive';
CREATE DATABASE metastore_db OWNER hive;

CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow_db OWNER airflow;
```

## MinIO bucket layout

Single bucket: `weather-lake`. Three prefixes, Hive-style partition naming:

```
weather-lake/
├── bronze/
│   └── region={region}/
│       └── dataset={archive|forecast}/
│           ├── year={YYYY}/lat={...}_lon={...}.json         (archive)
│           └── run_ts={YYYYMMDDTHHMMSSZ}/lat={...}_lon={...}.json  (forecast)
├── silver/
│   └── weather_observations/
│       └── dataset={archive|forecast}/date={YYYY-MM-DD}/
│           └── part-*.snappy.parquet
└── gold/
    └── weather_features/
        └── dataset={archive|forecast}/date={YYYY-MM-DD}/
            └── part-*.snappy.parquet
```

`key=value` directory names let Spark auto-discover partition columns when reading.

## File layout

```
data_engine/
├── README.md                       Front door
├── docker-compose.yml              Full stack (~14 services)
├── requirements.txt                Native venv deps
├── .env.example                    Config template
├── .gitignore
│
├── docs/
│   ├── PROGRESS.md                 Narrative walkthrough (for ML readers)
│   ├── USAGE.md                    Operational cookbook (every command)
│   └── WIKI.md                     ← you are here
│
├── ingestion/                      Python, runs in venv
│   ├── grid.py                     Armenia 10×10 grid generator
│   ├── schemas.py                  Pydantic models for Open-Meteo responses
│   ├── openmeteo_client.py         Async HTTPX client with bounded retries
│   ├── storage.py                  S3BronzeStorage (MinIO only — Phase 2a)
│   ├── coverage.py                 Gold-coverage manifest + missing-date math
│   ├── run_ingest.py               CLI: additive --backfill / --forecast / --force
│   └── healthcheck.py              Layered DNS/TCP/HTTP probe + CLI
│
├── spark_jobs/                     PySpark, runs inside spark-master
│   ├── bronze_to_silver.py         arrays_zip + explode + dedupe
│   ├── silver_to_gold.py           time + lag + rolling features
│   └── load_to_warehouse.py        Star schema load via JDBC
│
├── warehouse/
│   ├── ddl/                        Postgres DDL (numbered, run by hand)
│   │   ├── 001_dim_location.sql
│   │   ├── 002_dim_time.sql
│   │   ├── 003_fact_weather_observations.sql
│   │   ├── 004_predictions.sql        (Phase 2b)
│   │   └── 005_models.sql             (Phase 2b)
│   ├── client.py                   psycopg2 helpers (predictions + model registry)
│   ├── actuals_backfill.py         CLI: fill predictions.actual_value
│   └── hive/
│       └── create_external_tables.sql  Spark SQL DDL for silver+gold
│
├── ml/
│   ├── dataset.py                  load_gold (s3://) + WindowSpec + windowing
│   ├── models/lstm.py              Multivariate LSTM
│   ├── train.py                    CLI: train + versioned checkpoints + --resume
│   ├── evaluate.py                 CLI: backtest + per-horizon metrics
│   └── logging_utils.py            setup_logger + plot helpers
│
├── serving/
│   └── predictor.py                Predictor class (load + snap + predict + persist)
│
├── dashboard/                      Streamlit control plane (Phase 2c)
│   ├── app.py                      Entry — `streamlit run dashboard/app.py`
│   ├── state.py                    Cached predictor / coverage / models / SQL
│   ├── airflow_api.py              Thin REST wrapper for DAG triggers
│   └── pages/                      Status, Data, Train, Predict, Browse
│
├── airflow/
│   └── dags/weather_pipeline.py    @daily DAG: ingest → bronze/silver/gold →
│                                   warehouse → backfill_actuals → train_model
│
├── docker/                         Compose-related extras
│   ├── airflow/Dockerfile          Custom Airflow image (+ docker CLI + our deps)
│   ├── airflow/requirements-airflow.txt
│   ├── dashboard/Dockerfile        Streamlit image (slim + torch CPU + deps)
│   ├── dashboard/requirements-dashboard.txt
│   ├── spark/conf/spark-defaults.conf
│   ├── hive/conf/core-site.xml     S3A config for the metastore JVM
│   ├── hive/jdbc/postgresql-42.7.3.jar
│   └── postgres-init/*.sql
│
├── tests/                          Pytest unit tests
│
└── (gitignored, runtime)
    ├── .venv/
    ├── .env
    ├── checkpoints/                Versioned model + training/eval artifacts
    └── airflow/logs/               Per-task / scheduler logs
```

## Data schemas

### Bronze JSON

One file per (grid point × ingest window). Example:

```json
{
  "latitude": 41.3125,         // snapped to ERA5 0.25° grid
  "longitude": 46.625,
  "elevation": 989.0,
  "timezone": "UTC",
  "utc_offset_seconds": 0,
  "hourly_units": { "temperature_2m": "°C", ... },
  "hourly": {
    "time":                   ["2026-05-23T00:00", "2026-05-23T01:00", ...],
    "temperature_2m":         [8.9, 8.8, 8.7, ...],
    "relative_humidity_2m":   [82, 81, 81, ...],
    "dew_point_2m":           [6.0, 5.9, ...],
    "precipitation":          [0.0, 0.0, ...],
    "pressure_msl":           [1015.2, ...],
    "cloud_cover":            [40, 45, ...],
    "wind_speed_10m":         [7.1, 6.9, ...],
    "wind_direction_10m":     [275, 280, ...],
    "wind_gusts_10m":         [12.3, ...],
    "shortwave_radiation":    [0.0, 0.0, ...]
  }
}
```

10 hourly variables, parallel arrays (one value per timestamp).

### Silver / Gold Parquet

**Silver** (`s3a://weather-lake/silver/weather_observations`) — one row per (lat, lon, observed_at), 17 columns:

| Column | Type | Source |
|---|---|---|
| `region` | string | partition (path) |
| `dataset` | string | partition (path) — `archive` or `forecast` |
| `lat`, `lon` | double | bronze top-level |
| `elevation` | double | bronze top-level |
| `observed_at` | timestamp | parsed from `hourly.time[i]` |
| 10 weather vars (temperature_2m, …) | double | exploded from `hourly.<var>[i]` |
| `date` | date | derived `to_date(observed_at)` |

Partitioned by `(dataset, date)`. Deduped on `(region, dataset, lat, lon, observed_at)`.

**Gold** (`s3a://weather-lake/gold/weather_features`) — silver + 10 derived columns = 27 total:

| Added column | Type | Description |
|---|---|---|
| `hour` | int | 0–23 |
| `day_of_week` | int | 1=Sunday |
| `month` | int | 1–12 |
| `season` | string | `winter` / `spring` / `summer` / `autumn` |
| `is_weekend` | int | 0/1 |
| `temperature_2m_lag_1h` | double | `LAG(temperature_2m, 1) OVER (PARTITION BY lat,lon ORDER BY observed_at)` |
| `temperature_2m_lag_3h` | double | …lag 3 |
| `temperature_2m_lag_24h` | double | …lag 24 |
| `temperature_2m_roll24_mean` | double | 24-hour rolling mean |
| `temperature_2m_roll24_std` | double | 24-hour rolling std |

### Postgres star schema

`weather_dw.fact_weather_observations`:

```sql
CREATE TABLE fact_weather_observations (
    location_id           INTEGER NOT NULL REFERENCES dim_location (location_id),
    time_id               INTEGER NOT NULL REFERENCES dim_time     (time_id),
    dataset               TEXT    NOT NULL,
    temperature_2m        DOUBLE PRECISION,
    ... (9 other measures)
    PRIMARY KEY (location_id, time_id, dataset)
);
CREATE INDEX idx_fact_time     ON fact_weather_observations (time_id);
CREATE INDEX idx_fact_location ON fact_weather_observations (location_id);
CREATE INDEX idx_fact_dataset  ON fact_weather_observations (dataset);
```

`dim_location` (PK `location_id INTEGER`) — `region, lat, lon, elevation`, UNIQUE `(region, lat, lon)`.
`dim_time` (PK `time_id INTEGER`) — `observed_at UNIQUE, date, hour, day_of_week, month, year, season, is_weekend`.

Surrogate keys (INTEGER, not SERIAL) are assigned deterministically by `row_number()` in the Spark loader so a re-run from the same source data produces the same IDs.

Loader runs in **full-refresh mode**: single `TRUNCATE fact_weather_observations, dim_location, dim_time` statement (Postgres allows this without CASCADE when all referencing tables are listed), then `mode="append"`. Preserves indexes + FK constraints.

## Dashboard reference

Streamlit at http://localhost:8501. Five pages:

| Page | Module | What it does |
|---|---|---|
| Home | [dashboard/app.py](../dashboard/app.py) | 3-column summary: coverage, current model + stale-data warning, quickstart text |
| Status | [pages/1_Status.py](../dashboard/pages/1_Status.py) | Coverage, model registry table, recent Airflow runs |
| Data | [pages/2_Data.py](../dashboard/pages/2_Data.py) | Date-range picker → triggers `weather_pipeline` with `conf={start_date, end_date, force}` |
| Train | [pages/3_Train.py](../dashboard/pages/3_Train.py) | Versioned-checkpoint table + Plotly loss curves + "Train fresh" button |
| Predict | [pages/4_Predict.py](../dashboard/pages/4_Predict.py) | Forecast for a grid cell (persisted) + history tab with predicted-vs-actual overlay |
| Browse | [pages/5_Browse.py](../dashboard/pages/5_Browse.py) | Pre-canned SQL summaries + link to Adminer |

The dashboard imports `serving.predictor.Predictor` directly (no HTTP) — `@st.cache_resource` keeps it loaded across reruns. Postgres + MinIO access is cached with short TTLs; the sidebar's "Refresh caches" button forces a re-read.

Phase 2b added prediction persistence: every `/forecast` call writes `seq_out` rows to `weather_dw.predictions` (idempotent on `UNIQUE(model_version, location_id, prediction_made_at, target_time)`). Once observations land, the Airflow `backfill_actuals` task fills in `actual_value`.

## Airflow DAG reference

`airflow/dags/weather_pipeline.py`:

| Property | Value |
|---|---|
| `dag_id` | `weather_pipeline` |
| `schedule` | `@daily` |
| `start_date` | `2026-05-01` |
| `catchup` | `False` |
| `max_active_runs` | `1` |
| Default `retries` | `1` |
| Default `retry_delay` | `2 min` |
| Per-task `execution_timeout` | `30 min` |

Task graph (Phase 2a/2b):

```
ingest_archive → bronze_to_silver → silver_to_gold ┬→ load_warehouse → backfill_actuals
                                                   └→ train_model
```

| Task | Operator | Runs in | Command (simplified) |
|---|---|---|---|
| `ingest_archive` | BashOperator | airflow container | `python -m ingestion.run_ingest --backfill {ds-32}:{ds-2} --skip-check` (additive; params can override range + force) |
| `bronze_to_silver` | BashOperator | spark-master (via docker exec) | `spark-submit /opt/jobs/bronze_to_silver.py` |
| `silver_to_gold` | BashOperator | spark-master | `spark-submit /opt/jobs/silver_to_gold.py` |
| `load_warehouse` | BashOperator | spark-master | `spark-submit /opt/jobs/load_to_warehouse.py` |
| `backfill_actuals` | BashOperator | airflow container | `python -m warehouse.actuals_backfill` |
| `train_model` | BashOperator | airflow container | `python -m ml.train --gold s3://weather-lake/gold/weather_features --epochs 20` (writes a row to `models`) |

DAG params (settable from the Streamlit Data page or from the Airflow UI's "Trigger w/ config" button):

| Param | Type | Default |
|---|---|---|
| `start_date` | string (YYYY-MM-DD) | `""` → `logical_date − 32d` |
| `end_date` | string (YYYY-MM-DD) | `""` → `logical_date − 2d` (archive lag) |
| `force` | boolean | `false` |

The `docker exec`-style tasks work because the host's `/var/run/docker.sock` is bind-mounted into the Airflow scheduler container, and `docker.io` is installed in the custom Airflow image.

## GPU

The training stack auto-detects CUDA: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Confirmed working on **NVIDIA GeForce RTX 5070 Ti** (Blackwell, compute capability 12.0).

### Picking the torch wheel

| Wheel index | Supports |
|---|---|
| `https://download.pytorch.org/whl/cpu` | Any (CPU fallback) |
| `https://download.pytorch.org/whl/cu124` | Hopper and earlier (sm_90 and below) |
| `https://download.pytorch.org/whl/cu128` | Blackwell (sm_120) — needed for RTX 5070/5080/5090 |

Install / swap:

```powershell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

The serving layer also uses CUDA automatically if available.

## Git log + commit conventions

Phase-by-phase, one focused commit per slice:

```
Phase N[a/b/c]: <one-line summary>

<a few paragraphs explaining what, why, and how>
```

Multi-line commit messages are written via a HEREDOC (`git commit -m "$(cat <<'EOF' … EOF\n)"`) to dodge PowerShell parsing bugs.

## External resources

- **Open-Meteo docs**: https://open-meteo.com/en/docs (archive + forecast APIs, no key)
- **Apache Spark 3.5 docs**: https://spark.apache.org/docs/3.5.6/
- **Apache Hive 4.0 metastore**: https://hive.apache.org/docs/latest/
- **Airflow 2.10 docs**: https://airflow.apache.org/docs/apache-airflow/2.10.5/
- **PyTorch CUDA wheel index**: https://download.pytorch.org/whl/
- **Streamlit docs**: https://docs.streamlit.io/
- **Plotly Python**: https://plotly.com/python/
- **Adminer**: https://www.adminer.org/
- **MinIO client (mc) reference**: https://min.io/docs/minio/linux/reference/minio-mc.html
