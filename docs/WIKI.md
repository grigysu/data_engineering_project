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

### Ingestion / dashboard

| Var | Default | Used by |
|---|---|---|
| `INGEST_CONCURRENCY` | `5` | `ingestion.run_ingest` |
| `MODEL_CHECKPOINT` | `checkpoints/best.pt` | dashboard (checkpoint meta) + `ml.walk_forward` (default `--checkpoint`) |
| `GOLD_PATH` | `s3://weather-lake/gold/weather_features` | dashboard + `ml.walk_forward` (default `--gold`) |
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
| `weather_dw` | `weather` | Star schema (`fact_weather_observations`, `dim_location`, `dim_time`) + `predictions` (Phase 2b) + `backtest_groups` (Phase 2e). No model-registry table — model identity lives in the checkpoint dict. | DDL in [warehouse/ddl/](../warehouse/ddl/), run by hand |
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
├── config/
│   └── train.yaml                  ML hyperparameter config (Phase 2e single source of truth)
│
├── warehouse/
│   ├── ddl/                        Postgres DDL (numbered, run by hand)
│   │   ├── 001_dim_location.sql
│   │   ├── 002_dim_time.sql
│   │   ├── 003_fact_weather_observations.sql
│   │   ├── 004_predictions.sql           (Phase 2b)
│   │   └── 005_backtest_groups.sql       (Phase 2e — per-anchor MSE table)
│   ├── client.py                   psycopg2 helpers (predictions; no Postgres model registry)
│   ├── actuals_backfill.py         CLI: fill predictions.actual_value
│   ├── backtest_groups.py          UPSERT per-group MSE (matches walkforward:%)
│   └── hive/
│       └── create_external_tables.sql  Spark SQL DDL for silver+gold
│
├── ml/
│   ├── config.py                   Typed loader for config/train.yaml (Phase 2e)
│   ├── dataset.py                  load_gold (s3://) + WindowSpec + windowing
│   ├── models/lstm.py              Multivariate LSTM
│   ├── train.py                    CLI: train + versioned checkpoints + --cutoff-days + --resume
│   ├── walk_forward.py             CLI: daily DAG task — operational forecast + walk-forward eval
│   ├── evaluate.py                 CLI: per-horizon metrics + plots
│   └── logging_utils.py            setup_logger + plot helpers
│
├── dashboard/                      Streamlit control plane (Phase 2c)
│   ├── app.py                      Single-page entry — `streamlit run dashboard/app.py`
│   └── state.py                    Cached checkpoint meta + SQL helpers
│
├── airflow/
│   └── dags/weather_pipeline.py    @daily DAG: ingest → bronze/silver/gold →
│                                   warehouse → train_model → walk_forward → backfill_actuals
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

Streamlit at http://localhost:8501. **Single-page app** (Phase 2d collapsed the original multi-page layout): [dashboard/app.py](../dashboard/app.py) renders the region picker, snapshot combobox, backtest-groups table, predictions chart (snapshot overlays + backtest overlays), and a map view in one scroll.

Two surfaces let you put forecasts on the chart:

- **Snapshot combobox** — DAG snapshots ordered DESC by `prediction_made_at`. `ml.walk_forward` writes ~8 anchors per day (`lookback_days=7`, `stride_hours=24`), so the LIMIT-50 query shows ~6 days of operational forecasts. No model_version filter — walk_forward is the only writer of predictions.
- **Backtest-groups table** — per `(location, anchor)` best-MSE rows from [warehouse/backtest_groups.py](../warehouse/backtest_groups.py)'s UPSERT (matches `walkforward:%`). LIMIT 200 globally, sorted by MSE ASC. Check a row to add its forecast to the chart.

Caching: `dashboard/state.py` wraps Postgres reads and checkpoint-meta loads in `@st.cache_data`. The chart picks up new rows on the next Streamlit auto-rerun (TTL ~10s for SQL queries, ~60s for the checkpoint file).

Phase 2b added prediction persistence: predictions are idempotent on `UNIQUE(model_version, location_id, prediction_made_at, target_time)`. Phase 2e tags daily-DAG predictions as `walkforward:<orig_checkpoint_version>` so re-running the same checkpoint is a no-op.

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
| Default `execution_timeout` | `30 min` (overridden to `1 hr` on `walk_forward`) |

Task graph (Phase 2e):

```
ingest_archive → bronze_to_silver → silver_to_gold ┬→ load_warehouse ─┐
                                                   └→ train_model ────┴→ walk_forward → backfill_actuals
```

| Task | Operator | Runs in | Command (simplified) |
|---|---|---|---|
| `ingest_archive` | BashOperator | airflow container | `python -m ingestion.run_ingest --backfill {ds-32}:{ds-2} --skip-check` (additive; params can override range + force) |
| `bronze_to_silver` | BashOperator | spark-master (via docker exec) | `spark-submit /opt/jobs/bronze_to_silver.py` |
| `silver_to_gold` | BashOperator | spark-master | `spark-submit /opt/jobs/silver_to_gold.py` |
| `load_warehouse` | BashOperator | spark-master | `spark-submit /opt/jobs/load_to_warehouse.py` |
| `train_model` | BashOperator | airflow container | `python -m ml.train` — all hyperparams from [config/train.yaml](../config/train.yaml); trims gold to `[start, T − cutoff_days]` |
| `walk_forward` | BashOperator | airflow container | `python -m ml.walk_forward` — same checkpoint scored over `[T − lookback_days, T]` anchors; rightmost anchor is operational forecast; refreshes `backtest_groups`. `execution_timeout=1h`. |
| `backfill_actuals` | BashOperator | airflow container | `python -m warehouse.actuals_backfill` (catches anything else whose targets just landed) |

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

`ml.walk_forward` picks up the same device automatically when run inside the GPU-built Airflow scheduler.

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
