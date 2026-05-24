# Usage guide

Every command you need to run the project, organised by workflow. PowerShell syntax (the project is built on Windows + Docker Desktop), but most commands are bash-compatible — see the [shell notes at the end](#shell-notes).

## Table of contents

- [First-time setup](#first-time-setup)
- [Bring up / tear down the stack](#bring-up--tear-down-the-stack)
- [Workflow 1 — ingest fresh data](#workflow-1--ingest-fresh-data)
- [Workflow 2 — run the Spark ETL chain](#workflow-2--run-the-spark-etl-chain)
- [Workflow 3 — query the lake / warehouse](#workflow-3--query-the-lake--warehouse)
- [Workflow 4 — train + evaluate the model](#workflow-4--train--evaluate-the-model)
- [Workflow 5 — serve forecasts over HTTP](#workflow-5--serve-forecasts-over-http)
- [Workflow 6 — orchestrate everything via Airflow](#workflow-6--orchestrate-everything-via-airflow)
- [Run quality gates before committing](#run-quality-gates-before-committing)
- [Troubleshooting](#troubleshooting)
- [Shell notes](#shell-notes)

---

## First-time setup

```powershell
# 1. Python venv with all deps
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Local config (defaults work; edit if you want to change Region, etc.)
Copy-Item .env.example .env

# 3. Bring up the stack (first pull ≈ 3 GB)
docker compose up -d

# 4. Wait for postgres + minio + hive_init + spark_init to all go healthy
docker compose ps
```

After step 4 you should see `healthy` next to `minio` and `postgres`, and `Exited (0)` next to the one-shot init containers (`minio_init`, `spark_init`, `hive_init`, `airflow_init`).

### One-time DDL bootstrap

Two pieces of DDL need to run once after the stack is up — they're not in the auto-init because they depend on the existing data:

```powershell
# Hive metastore: register silver + gold parquet as external tables
docker cp warehouse/hive/create_external_tables.sql weather_spark_master:/tmp/
docker exec weather_spark_master /opt/spark/bin/spark-sql `
  --master spark://spark-master:7077 -f /tmp/create_external_tables.sql

# Postgres warehouse: create the star schema
Get-ChildItem warehouse/ddl/*.sql | Sort-Object Name | ForEach-Object {
    Get-Content $_.FullName -Raw | docker exec -i weather_postgres `
        psql -U weather -d weather_dw -v ON_ERROR_STOP=1
}
```

---

## Bring up / tear down the stack

```powershell
docker compose up -d           # bring everything up
docker compose ps              # what's running, status, ports
docker compose logs -f minio   # tail one service's logs
docker compose stop            # stop containers (keep volumes — fast restart)
docker compose down            # stop + remove containers (keep volumes)
docker compose down -v         # NUKE: also wipes minio/postgres/ivy volumes
```

Restarting just one service after a config change:

```powershell
docker compose restart spark-master spark-worker
docker compose up -d --force-recreate hive-metastore   # picks up new env vars
```

---

## Workflow 1 — ingest fresh data

### Healthcheck (preflight — always run first if archive endpoint is iffy)

```powershell
python -m ingestion.healthcheck                        # human-readable
python -m ingestion.healthcheck --json                 # machine-readable
python -m ingestion.healthcheck --require forecast     # only forecast must be OK
```

Exit code: `0` if all required endpoints OK, `1` otherwise.

### Pull the 7-day forecast for every grid point

```powershell
python -m ingestion.run_ingest --forecast --forecast-days 7
```

Uses `STORAGE_BACKEND` from `.env` (default `local` → `./data/lake/bronze/...`). For MinIO:

```powershell
$env:STORAGE_BACKEND='s3'
python -m ingestion.run_ingest --forecast --forecast-days 7
```

### Backfill historical data (when the archive endpoint recovers)

```powershell
python -m ingestion.run_ingest --backfill 2023-01-01:2025-12-31 --concurrency 8
```

100 grid points × 3 years = 300 API calls; chunked by year per point. The preflight will refuse if archive endpoint isn't OK — pass `--skip-check` to override.

### CLI flags reference

| Flag | Default | Notes |
|---|---|---|
| `--backfill START:END` | — | `YYYY-MM-DD:YYYY-MM-DD` historical window |
| `--forecast` | — | Pull latest forecast |
| `--forecast-days N` | 14 | Forecast horizon (Open-Meteo max 16) |
| `--concurrency N` | 5 | Max in-flight HTTP requests |
| `--skip-check` | off | Skip the preflight Open-Meteo healthcheck |

---

## Workflow 2 — run the Spark ETL chain

All three jobs run inside the `weather_spark_master` container, against MinIO via S3A.

```powershell
# Bronze → Silver: parse + explode + dedupe → partitioned Parquet
docker exec weather_spark_master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 /opt/jobs/bronze_to_silver.py

# Silver → Gold: time, lag, rolling features
docker exec weather_spark_master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 /opt/jobs/silver_to_gold.py

# Gold → Postgres warehouse (truncate + insert, preserves FKs)
docker exec weather_spark_master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 /opt/jobs/load_to_warehouse.py
```

### CLI flags

Each job accepts `--bronze`, `--silver`, `--gold`, `--jdbc-url`, etc. Defaults match the docker-compose service names. See `python /opt/jobs/<file>.py --help` inside the container.

---

## Workflow 3 — query the lake / warehouse

### Via Spark SQL (lake / Hive)

Interactive REPL inside the spark-master container:

```powershell
docker exec -it weather_spark_master /opt/spark/bin/spark-sql `
  --master spark://spark-master:7077
```

Then:

```sql
SHOW DATABASES;
SELECT COUNT(*) FROM gold.weather_features;
SELECT dataset, date, COUNT(*) FROM gold.weather_features GROUP BY dataset, date;
```

Run a SQL file end-to-end:

```powershell
docker cp queries.sql weather_spark_master:/tmp/queries.sql
docker exec weather_spark_master /opt/spark/bin/spark-sql `
  --master spark://spark-master:7077 -f /tmp/queries.sql
```

### Via psql (warehouse / star schema)

```powershell
docker exec -it weather_postgres psql -U weather -d weather_dw
```

Sample analytical query (Top-N hottest grid cells):

```sql
SELECT l.lat, l.lon, MAX(f.temperature_2m) AS hottest
FROM   fact_weather_observations f
JOIN   dim_location l USING (location_id)
GROUP  BY l.lat, l.lon
ORDER  BY hottest DESC;
```

### Via DBeaver / Metabase / any JDBC tool

- **Postgres warehouse**: `jdbc:postgresql://localhost:5432/weather_dw`, user `weather`, pass `weather`
- **Hive Metastore**: `thrift://localhost:9083` (BI tools can use this to enumerate tables)

---

## Workflow 4 — train + evaluate the model

### Sync gold to local disk (optional — train can also read directly from MinIO)

```powershell
docker run --rm --entrypoint /bin/sh --network data_engine_default `
  -v ${PWD}/data:/host_data minio/mc:latest -c `
  "mc alias set local http://minio:9000 minioadmin minioadmin > /dev/null && `
   mc mirror --overwrite local/weather-lake/gold/weather_features/ /host_data/lake/gold/weather_features/"
```

### Train

```powershell
# Local sync'd gold
python -m ml.train --gold data/lake/gold/weather_features --epochs 30

# OR straight from MinIO (no local sync needed)
$env:S3_ENDPOINT='http://localhost:9000'
$env:S3_ACCESS_KEY='minioadmin'
$env:S3_SECRET_KEY='minioadmin'
python -m ml.train --gold s3://weather-lake/gold/weather_features --epochs 30
```

Artifacts land in `checkpoints/`:
- `best.pt` — model weights + spec + feature stats
- `best.metrics.json` — summary
- `train.log` — full timestamped console output
- `training_log.csv` — per-epoch (epoch, train_mse, val_mse, is_best)
- `loss_curves.png` — train + val MSE chart

### Evaluate (backtest the saved checkpoint)

```powershell
python -m ml.evaluate --gold data/lake/gold/weather_features
```

Additional artifacts:
- `eval.log` — timestamped output
- `per_horizon_metrics.csv` — h_ahead, MAE, RMSE, MAPE
- `per_horizon_error.png` — MAE/RMSE bars + persistence baseline
- `predictions_vs_actual.png` — 6 sample windows truth-vs-forecast

### Training CLI flags

| Flag | Default |
|---|---|
| `--gold PATH` | required |
| `--seq-in` | 24 (hours of context) |
| `--seq-out` | 6 (hours to predict) |
| `--batch-size` | 32 |
| `--epochs` | 20 |
| `--lr` | 1e-3 |
| `--hidden` | 64 (LSTM hidden size) |
| `--layers` | 2 (LSTM layers) |
| `--val-fraction` | 0.2 (time-based split) |
| `--seed` | 42 |
| `--checkpoint` | `checkpoints/best.pt` |

---

## Workflow 5 — serve forecasts over HTTP

```powershell
# Start the server (foreground)
uvicorn serving.api:app --host 0.0.0.0 --port 8000

# OR in another window in the background
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --log-level warning
```

Hit it:

```powershell
# Liveness + model summary
curl http://localhost:8000/health

# Which grid points do we have data for?
curl http://localhost:8000/grid_points

# Forecast for an arbitrary (lat, lon) — snaps to nearest known grid cell
curl -X POST http://localhost:8000/forecast `
  -H "Content-Type: application/json" `
  -d '{"lat": 40.18, "lon": 44.51}'

# Swagger UI
start http://localhost:8000/docs
```

### Config (env vars)

| Var | Default |
|---|---|
| `MODEL_CHECKPOINT` | `checkpoints/best.pt` |
| `GOLD_PATH` | `data/lake/gold/weather_features` (local) or `s3://...` |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` | MinIO defaults |

---

## Workflow 6 — orchestrate everything via Airflow

### Web UI

http://localhost:8081 — login `admin` / `admin`. The `weather_pipeline` DAG runs `@daily` (catchup off, max 1 active run).

### CLI

```powershell
# List DAGs (also catches import errors)
docker exec weather_airflow_scheduler airflow dags list

# Force a fresh parse / serialization (if you just edited the DAG file)
docker exec weather_airflow_scheduler airflow dags reserialize

# Manually trigger
docker exec weather_airflow_scheduler airflow dags trigger weather_pipeline

# Poll task states for a specific run
docker exec weather_airflow_scheduler airflow tasks states-for-dag-run `
  weather_pipeline manual__2026-05-24T09:15:26+00:00

# Tail a task's log (paths are inside the container's /opt/airflow/logs/)
docker exec weather_airflow_scheduler bash -c "ls /opt/airflow/logs/dag_id=weather_pipeline/run_id=*/task_id=train_model/"
```

### Editing the DAG

Edit `airflow/dags/weather_pipeline.py` on the host — it's bind-mounted into the scheduler. The scheduler re-parses every ~30s. To force immediately:

```powershell
docker exec weather_airflow_scheduler airflow dags reserialize
```

---

## Run quality gates before committing

Always run these four before `git commit`:

```powershell
. .\.venv\Scripts\Activate.ps1
ruff check .                                    # lint (auto-fix: ruff check --fix .)
ruff format --check .                           # format (auto-fix: ruff format .)
pytest tests/ -q                                # tests (currently 26)
python -c "import ingestion, ingestion.grid, ingestion.schemas, ingestion.openmeteo_client, `
  ingestion.storage, ingestion.run_ingest, ingestion.healthcheck, ml, ml.dataset, ml.train, `
  ml.evaluate, ml.logging_utils, ml.models.lstm, serving, serving.api, serving.predictor; `
  print('venv imports OK')"
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `docker: not found` after restart | Windows: PATH doesn't propagate to running processes. Restart Claude Code / your shell. |
| `Cannot find pipe dockerDesktopLinuxEngine` | Docker Desktop is not running. Launch it, wait for whale icon to go steady. |
| Open-Meteo backfill returns 504s | Upstream `archive-api.open-meteo.com` is intermittently down. Run `python -m ingestion.healthcheck` to confirm. Forecast endpoint usually still healthy. |
| Spark `_SUCCESS` written but no data files | Cloud committer is mis-configured. We use the default `FileOutputCommitter`; if you re-enabled the directory committer, swap it back (see `docker/spark/conf/spark-defaults.conf` history). |
| Spark `--packages` JARs fail with `FileNotFoundException` in `~/.ivy2/cache` | Named volume created as `root:root`. The `spark_init` one-shot container fixes this; if it didn't run, `docker exec --user root weather_spark_master chown -R spark:spark /home/spark/.ivy2`. |
| Hive `ClassNotFoundException: S3AFileSystem` | `HADOOP_OPTIONAL_TOOLS=hadoop-aws` isn't reaching the metastore. Verify the compose env block. |
| `Failed to create external path s3a://.../warehouse/silver.db` on `CREATE DATABASE` | The metastore is trying to `mkdir` the default LOCATION. `spark.sql.warehouse.dir` should be `file:/tmp/spark-warehouse` (local), not s3a://. |
| `MSCK REPAIR TABLE silver.weather_observations` returns 0 rows | The Hive table was created but partitions weren't scanned. Re-run MSCK REPAIR; or check `Partition Provider: Catalog` in `DESCRIBE EXTENDED`. |
| Airflow DAG triggered but stays "queued" | Scheduler hasn't picked it up yet. `docker exec weather_airflow_scheduler airflow dags reserialize`. |
| `DagNotFound: Dag id weather_pipeline not found in DagModel` | Same as above — the file is parsed but not in the DB yet. Reserialize. |
| `BashOperator` fails with `cannot connect to the Docker daemon` | The `docker.sock` bind-mount didn't work. Verify `/var/run/docker.sock:/var/run/docker.sock` is in the airflow_scheduler service. |
| Training: "zero windows produced" | Not enough rows per location for `seq_in + seq_out`. Backfill more data, or reduce `--seq-in`. |
| Training: GPU not used (says `device=cpu`) | Wrong torch wheel. `pip uninstall torch` then `pip install torch --index-url https://download.pytorch.org/whl/cu128` (or cu124 for non-Blackwell). |
| Postgres `permission denied` for `hive` or `airflow` user | Re-run the `CREATE USER` / `GRANT` SQL manually (see [docs/WIKI.md#postgres-databases](WIKI.md#postgres-databases)). |

---

## Shell notes

- The project is developed on **PowerShell** (Windows + Docker Desktop). All `python -m ...` commands also work in bash / zsh — just replace `. .\.venv\Scripts\Activate.ps1` with `source .venv/Scripts/activate`.
- PowerShell-specific syntax used above:
  - `${PWD}` instead of `$PWD` in interpolated strings
  - Backtick `` ` `` for line continuation instead of `\`
  - `$env:VAR` for env vars instead of `VAR=...`
- For multi-line SQL commands, prefer `docker cp` + `spark-sql -f` over piping into stdin — PowerShell sometimes injects a UTF-8 BOM that breaks the first statement.
