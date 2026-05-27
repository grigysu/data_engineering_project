# Usage guide

Every command you need to run the project, organised by workflow. PowerShell syntax (the project is built on Windows + Docker Desktop), but most commands are bash-compatible — see the [shell notes at the end](#shell-notes).

## Table of contents

- [First-time setup](#first-time-setup)
- [Bring up / tear down the stack](#bring-up--tear-down-the-stack)
- [Workflow 1 — ingest fresh data](#workflow-1--ingest-fresh-data)
- [Workflow 2 — run the Spark ETL chain](#workflow-2--run-the-spark-etl-chain)
- [Workflow 3 — query the lake / warehouse](#workflow-3--query-the-lake--warehouse)
- [Workflow 4 — train + evaluate the model](#workflow-4--train--evaluate-the-model)
- [Workflow 5 — drive the pipeline from the Streamlit dashboard](#workflow-5--drive-the-pipeline-from-the-streamlit-dashboard)
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

# Postgres warehouse: create the star schema + (Phase 2b) predictions + models tables
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

> **Phase 2a note**: MinIO is now the only storage backend. There is no `STORAGE_BACKEND` switch and no local `data/` folder. Make sure `docker compose up -d` is running before ingestion.

### Backfill historical archive (additive)

```powershell
python -m ingestion.run_ingest --backfill 2026-05-01:2026-05-24
```

This is **additive**: a gold-coverage manifest at `s3://weather-lake/manifest/ingested.json` records which dates have already been ingested, so re-running the same range is a no-op and extending it only fetches the gap. 100 grid points × N years = N×100 API calls, chunked by year per point.

```powershell
# Force re-fetch even for already-covered chunks:
python -m ingestion.run_ingest --backfill 2026-05-01:2026-05-24 --force
```

### Pull the latest 14-day forecast for every grid point

```powershell
python -m ingestion.run_ingest --forecast --forecast-days 14
```

### CLI flags reference

| Flag | Default | Notes |
|---|---|---|
| `--backfill START:END` | — | `YYYY-MM-DD:YYYY-MM-DD` historical window (additive) |
| `--forecast` | — | Pull latest forecast |
| `--forecast-days N` | 14 | Forecast horizon (Open-Meteo max 16) |
| `--concurrency N` | 5 | Max in-flight HTTP requests |
| `--skip-check` | off | Skip the preflight Open-Meteo healthcheck |
| `--force` | off | Ignore the coverage manifest; re-fetch every chunk |

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

Phase 2e replaced the old "train once, predict the next 24h from `now`" path with **walk-forward**: train on gold up to `T − cutoff_days`, then in one inference pass score the model against the last `lookback_days` of anchors *and* produce the operational forecast (the rightmost anchor whose targets are still in the future). The DAG runs both pieces daily; the same modules also run by hand.

### Hyperparameter config — `config/train.yaml`

Single source of truth. The Airflow DAG, the Makefile, and both CLIs read this; nothing is duplicated in bash strings.

```yaml
train:
  seq_in: 48
  seq_out: 24
  hidden_size: 64
  num_layers: 2
  epochs: 5
  batch_size: 16
  learning_rate: 0.001
  val_fraction: 0.2
  seed: 42
  cutoff_days: 7      # reserve the last N days as honest holdout

backtest:
  lookback_days: 7    # walk_forward anchors span [T - N .. T]
  stride_hours: 24    # one anchor per day

paths:
  gold: "s3://weather-lake/gold/weather_features"
  checkpoint_dir: "checkpoints"
```

Inspect the resolved config (YAML + dataclass defaults) at any time:

```powershell
make show-config
```

CLI flags on `ml.train` and `ml.walk_forward` override individual YAML values per-run — anything you don't pass uses the YAML.

### Train (fresh run)

```powershell
$env:S3_ENDPOINT='http://localhost:9000'
$env:S3_ACCESS_KEY='minioadmin'
$env:S3_SECRET_KEY='minioadmin'
python -m ml.train                          # all hyperparams from YAML
python -m ml.train --epochs 5 --lr 5e-4     # one-off overrides
python -m ml.train --cutoff-days 0          # single-shot: train on ALL gold
```

Reads gold parquet directly from MinIO (no local sync). `cutoff_days` trims the most recent N days off the training set so `walk_forward` has actuals to score against — pass `--cutoff-days 0` for single-shot training (useful for ad-hoc experiments where you don't need the holdout).

Writes versioned artifacts to `checkpoints/`:

- `<version>.pt` — weights + spec + feature stats + provenance (data range, row count, hyperparams, optimizer state)
- `<version>.metrics.json` — summary
- `<version>_log.csv` — per-epoch (epoch, train_mse, val_mse, is_best)
- `<version>_loss.png` — train + val MSE chart
- `best.pt` — copy of the latest run (used by `ml.walk_forward` and the dashboard)
- `train.log` — timestamped console output

`<version>` is an ISO-8601 timestamp (Windows-safe filename, e.g. `2026-05-27T13-15-22Z`). Model identity lives entirely in the checkpoint dict — there's no Postgres model-registry table.

### Continue training from a checkpoint

```powershell
python -m ml.train --resume checkpoints/best.pt --extra-epochs 5
```

Loads the previous state dict + optimizer state and continues from `epoch = checkpoint["epoch"] + 1`. Refuses if `feature_columns` or hidden/layers differ — schema-drift guard.

### Walk-forward (operational forecast + honest 7-day evaluation, one pass)

```powershell
python -m ml.walk_forward                    # uses checkpoints/best.pt + YAML
python -m ml.walk_forward --lookback-days 14 --stride-hours 6
```

For each cell, enumerates anchors in `[T − lookback_days .. T]` at `stride_hours` intervals and persists `seq_out` predictions per anchor. The rightmost anchor (T) is the operational forecast — its targets extend into the future, so `actual_value` stays NULL until subsequent runs backfill from new observations. Anchors with `target_time` already in the past get filled immediately via `backfill_actuals_for_version`. Predictions land in `predictions` tagged `model_version = walkforward:<orig_checkpoint_version>` and the same task refreshes `backtest_groups` so the dashboard's BT table picks them up.

Re-running with the same checkpoint is a no-op (UNIQUE constraint on `(model_version, location_id, prediction_made_at, target_time)`). A fresh training produces a new `orig_version` → fresh `walkforward:*` rows.

### Evaluate (per-horizon metrics from a checkpoint)

```powershell
python -m ml.evaluate --gold s3://weather-lake/gold/weather_features
```

Artifacts:
- `eval.log` — timestamped output
- `per_horizon_metrics.csv` — h_ahead, MAE, RMSE, MAPE
- `per_horizon_error.png` — MAE/RMSE bars + persistence baseline
- `predictions_vs_actual.png` — 6 sample windows truth-vs-forecast

### CLI flags reference

All hyperparam flags below default to the YAML value if omitted (sentinel: `None` in argparse). The YAML in turn falls back to the dataclass defaults in [ml/config.py](../ml/config.py).

**`python -m ml.train`**

| Flag | Source if omitted |
|---|---|
| `--config PATH` | `config/train.yaml` |
| `--gold` | `paths.gold` |
| `--seq-in`, `--seq-out` | `train.seq_in` / `train.seq_out` |
| `--batch-size`, `--epochs`, `--lr` | corresponding `train.*` |
| `--hidden`, `--layers` | `train.hidden_size` / `train.num_layers` |
| `--val-fraction`, `--seed` | `train.val_fraction` / `train.seed` |
| `--cutoff-days N` | `train.cutoff_days`; `0` = single-shot, train on all gold |
| `--checkpoint-dir` | `paths.checkpoint_dir` |
| `--resume PATH` | — | continue from an existing checkpoint |
| `--extra-epochs N` | `0` | epochs to add on top of the resume point |

**`python -m ml.walk_forward`**

| Flag | Source if omitted |
|---|---|
| `--config PATH` | `config/train.yaml` |
| `--checkpoint PATH` | `<paths.checkpoint_dir>/best.pt` |
| `--gold` | `paths.gold` |
| `--lookback-days N` | `backtest.lookback_days` |
| `--stride-hours N` | `backtest.stride_hours` |

---

## Workflow 5 — drive the pipeline from the Streamlit dashboard

> Phase 2c replaced FastAPI with a single Streamlit control plane. The dashboard runs as a Docker service (`weather_dashboard`) and is the recommended way to operate the pipeline.

### Open the dashboard

```powershell
docker compose up -d dashboard
start http://localhost:8501
```

Or run natively from an activated venv. Streamlit only adds the script's own directory to `sys.path`, so the repo root needs to be on `PYTHONPATH` for `from dashboard.state import …` to resolve:

```powershell
$env:PYTHONPATH = $PWD
streamlit run dashboard/app.py --server.port 8501
```

### Pages

| Page | What it does |
|---|---|
| **Status** | Coverage summary, registered models table, recent Airflow run states |
| **Data** | Pick a date range → "Backfill missing days" triggers the Airflow DAG with conf |
| **Train** | Browse versioned checkpoints, plot loss curves with Plotly, kick off the training DAG |
| **Predict** | Pick a grid cell → forecast (persisted to Postgres); second tab plots historical predicted vs. actual |
| **Browse** | Pre-canned warehouse SQL summaries + a link out to Adminer |

### Config (env vars consumed by the dashboard)

| Var | Default |
|---|---|
| `MODEL_CHECKPOINT` | `checkpoints/best.pt` |
| `GOLD_PATH` | `s3://weather-lake/gold/weather_features` |
| `CHECKPOINT_DIR` | `checkpoints` |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` | MinIO defaults |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | docker-compose defaults |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USER`, `AIRFLOW_PASSWORD` | `http://localhost:8081`, `admin`, `admin` |
| `WEATHER_DAG_ID` | `weather_pipeline` |
| `ADMINER_URL` | `http://localhost:8082` |

### Browse the warehouse directly (Adminer)

```powershell
start http://localhost:8082
```

Server: `postgres` (use that hostname from the Adminer container). User `weather`, DB `weather_dw`, password from `.env`.

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
docker exec weather_airflow_scheduler bash -c "ls /opt/airflow/logs/dag_id=weather_pipeline/run_id=*/task_id=walk_forward/"
```

### Task graph

```
ingest_archive → bronze_to_silver → silver_to_gold ┬→ load_warehouse ─┐
                                                   └→ train_model ────┴→ walk_forward → backfill_actuals
```

| Task | What it does |
|---|---|
| `ingest_archive` | Additive Open-Meteo archive pull → MinIO bronze |
| `bronze_to_silver`, `silver_to_gold` | Spark medallion ETL |
| `load_warehouse` | Spark JDBC into Postgres star schema |
| `train_model` | `python -m ml.train` — reads `config/train.yaml`, trims gold to `[start, T − cutoff_days]`, writes `checkpoints/best.pt` |
| `walk_forward` | `python -m ml.walk_forward` — same model evaluated over `[T − lookback_days, T]` anchors; rightmost anchor = operational forecast; refreshes `backtest_groups` |
| `backfill_actuals` | Fills `predictions.actual_value` for any rows whose observation has since landed (catches non-walk_forward versions too) |

`walk_forward` has its `execution_timeout` bumped to 1 hour (others use the 30-min `default_args` value).

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
pytest tests/ -q                                # tests
python -c "import ingestion.run_ingest, ingestion.coverage, ingestion.storage, ingestion.healthcheck, `
  ml.config, ml.dataset, ml.train, ml.evaluate, ml.walk_forward, ml.models.lstm, `
  warehouse.client, warehouse.actuals_backfill, warehouse.backtest_groups, `
  dashboard.state; `
  print('venv imports OK')"
```

Or use the Makefile shortcuts:

```powershell
make check          # ruff check + format check + pytest (mirrors the pre-commit gate)
make show-config    # print the resolved config/train.yaml as JSON
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
