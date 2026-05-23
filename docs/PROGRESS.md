# Progress so far — a walkthrough for AI engineers

This is a tour of what's been built across **Phases 0–3** of the project, written for ML/AI folks who are comfortable with PyTorch and pandas but haven't worked much with SQL warehouses or data-engineering tools like Spark, Hive, or S3.

If you only read one section, read [The big picture](#the-big-picture).

---

## Table of contents

- [What this project is](#what-this-project-is)
- [The big picture](#the-big-picture)
- [Quick glossary for ML folks](#quick-glossary-for-ml-folks)
- [Phase 1 — Ingestion (getting the data in)](#phase-1--ingestion-getting-the-data-in)
- [Phase 2a — Spark medallion ETL (cleaning + features)](#phase-2a--spark-medallion-etl-cleaning--features)
- [Phase 2b — Hive Metastore (SQL access to the lake)](#phase-2b--hive-metastore-sql-access-to-the-lake)
- [Phase 3 — Postgres star schema (analytics warehouse)](#phase-3--postgres-star-schema-analytics-warehouse)
- [Where the model fits next (Phase 4)](#where-the-model-fits-next-phase-4)
- [Running the whole thing](#running-the-whole-thing)

---

## What this project is

We're building an **end-to-end pipeline that pulls weather data, transforms it, and feeds a PyTorch sequence model that forecasts future temperatures over Armenia** (a 10×10 grid of locations).

The point isn't really the model — most weather forecasts are made by huge physics simulators (NWP models) that an LSTM trained on 3 years of data has no chance of beating. The point is to build a **realistic, portfolio-grade data pipeline** that demonstrates the modern data stack the way a real production system would use it.

You can think of it as: "all the preprocessing infrastructure that real ML systems need, but usually hand-waved away in a Kaggle notebook."

---

## The big picture

```
┌──────────────┐   1. INGEST    ┌──────────────┐
│  Open-Meteo  │ ───────────▶   │   bronze/    │  raw JSON, one file per
│   (free API) │   Python +     │  (MinIO,     │  (region, dataset, time-window,
└──────────────┘   httpx        │   S3-like)   │   grid point)
                                └──────┬───────┘
                                       │ 2. CLEAN + EXPLODE
                                       │   (Spark)
                                       ▼
                                ┌──────────────┐
                                │   silver/    │  Parquet, one row per
                                │  (MinIO)     │  (lat, lon, observed_at),
                                └──────┬───────┘  17 cols, partitioned by date
                                       │ 3. FEATURE ENGINEERING
                                       │   (Spark: time, lag, rolling)
                                       ▼
                                ┌──────────────┐
                                │   gold/      │  Parquet, 27 cols
                                │  (MinIO)     │  ← THIS is what PyTorch reads
                                └──────┬───────┘
                                       │ 4. LOAD TO WAREHOUSE
                                       │   (Spark → JDBC)
                                       ▼
                                ┌──────────────────────────┐
                                │  Postgres (weather_dw):  │  Star schema for
                                │  • dim_location          │  human-driven SQL
                                │  • dim_time              │  analytics + BI
                                │  • fact_weather_observations │  tools.
                                └──────────────────────────┘

           ┌─────────────────────────────────────────┐
           │  Hive Metastore (separate from Postgres │  Records "this Parquet
           │  warehouse data — uses its own DB):     │  is a SQL table called
           │  silver.weather_observations            │  X." Lets BI tools and
           │  gold.weather_features                  │  spark-sql query the
           └─────────────────────────────────────────┘  lake by name.
```

Three planes of storage, three different audiences:

| Layer | Format | Who queries it |
|---|---|---|
| **Lake** (bronze/silver/gold) | Parquet on MinIO | Spark jobs, PyTorch |
| **Hive-cataloged lake** | Same Parquet, registered as SQL tables | BI tools, ad-hoc analysts via `spark-sql` |
| **Warehouse** (Postgres star schema) | Relational tables | Analysts, dashboards, anyone who speaks SQL |

This separation matters because **the data lake is optimized for the model** (columnar Parquet, partitioned by date, lots of feature columns), while **the warehouse is optimized for humans** (proper normalized schema, indexed joins, fast `GROUP BY`).

---

## Quick glossary for ML folks

| Term | What it actually is for ML people |
|---|---|
| **Parquet** | A binary, columnar file format. Think of it as "a `.npy` file but with schema, compression, and column-level random access." Spark and pandas both read it natively. |
| **MinIO** | An S3-compatible object store running locally. We use it instead of writing files directly to disk because it gives us the same API (`s3a://...` URIs, `boto3` client) that AWS S3 does — so the code is cloud-portable without rewriting. |
| **medallion architecture** | A convention for naming layers in a data lake: **bronze** = raw (untouched API responses), **silver** = cleaned + typed + deduped, **gold** = feature-engineered, ready for model or BI. It's basically: "raw / clean / model-ready." Same idea as `data/raw/` `data/processed/` `data/features/` in a Kaggle project, just standardized. |
| **Spark / PySpark** | A distributed dataframe library. Looks like pandas but the data lives across many workers and operations are lazy (build a query plan, execute on `.show()` / `.count()` / `.write`). Use it when your data doesn't fit in RAM, or when you want to demonstrate you can. For our scale it's overkill but it's the right tool to learn. |
| **JDBC** | Just the standard way Java apps talk to SQL databases. Spark uses it to read/write Postgres. The Python equivalent is `psycopg2` or `SQLAlchemy`. |
| **Hive Metastore** | A separate little service whose only job is to remember: "the parquet files at `s3a://weather-lake/gold/weather_features/` are a SQL table called `gold.weather_features` with these 27 columns and partitions." Without it, every tool that wants to query the lake needs to figure out the schema and partition layout itself. With it, any SQL tool (BI dashboards, DBeaver, `spark-sql`) can `SELECT * FROM gold.weather_features` directly. |
| **External table** | A Hive table whose data lives somewhere else (in our case, Parquet on MinIO). The metastore only stores metadata — schema, partition info, location. Dropping the table doesn't delete the underlying files. |
| **Star schema** | A relational design pattern for analytics. One central **fact** table (rows = measurements), surrounded by **dimension** tables (rows = the things you measured — locations, time, etc.). Fact rows reference dimensions by ID. Optimized for `GROUP BY`-heavy queries like "average temperature by month and location." If you've seen "wide vs long format" in pandas — star schema is the relational version of "long format with normalized lookup tables." |
| **Surrogate key** | An artificial integer ID we generate for each row in a dim table. The "natural key" for a location is `(region, lat, lon)`, but joining on three columns is annoying — so we assign each location a unique `location_id` and use that everywhere. |
| **Partition** (in Parquet) | Subdirectories named like `dataset=forecast/date=2026-05-23/` that let the query engine skip whole chunks of data when filtering. Equivalent to indexing on those columns at the filesystem level. |
| **Docker Compose** | Define multiple containers in one YAML file, bring them all up with one command. We use it to spin up MinIO + Postgres + Spark + Hive Metastore as a coordinated local stack. |

---

## Phase 1 — Ingestion (getting the data in)

**Goal:** pull weather data from a free API and land it in an object store as "untouched as possible," partitioned by useful keys.

### Data source

[Open-Meteo](https://open-meteo.com) — free, no API key needed. Two endpoints we use:
- **Forecast** (`api.open-meteo.com/v1/forecast`): up to 16 days ahead, hourly. Healthy.
- **Archive** (`archive-api.open-meteo.com/v1/archive`): historical hourly back ~80 years, sourced from ERA5 reanalysis. ⚠️ Currently down upstream (504s); we have a healthcheck that detects this and aborts cleanly so we don't burn 400 retries.

Each response looks like:
```json
{
  "latitude": 41.3125, "longitude": 46.625, "elevation": 989.0,
  "hourly": {
    "time": ["2026-05-23T00:00", "2026-05-23T01:00", ...],
    "temperature_2m": [8.9, 8.8, 8.7, ...],
    "wind_speed_10m": [7.1, 6.9, ...],
    ... (10 weather variables total)
  }
}
```

### What ships

| File | Role |
|---|---|
| [ingestion/grid.py](../ingestion/grid.py) | Generates the 10×10 lat/lon grid covering Armenia (configurable via env vars). |
| [ingestion/schemas.py](../ingestion/schemas.py) | Pydantic models for the Open-Meteo response — catches schema drift at ingestion time. |
| [ingestion/openmeteo_client.py](../ingestion/openmeteo_client.py) | Async HTTPX client with bounded retries (only 429 + 5xx). |
| [ingestion/storage.py](../ingestion/storage.py) | Swappable `BronzeStorage` protocol — `LocalBronzeStorage` for dev, `S3BronzeStorage` (boto3) for MinIO. Switch with `STORAGE_BACKEND=local\|s3`. |
| [ingestion/run_ingest.py](../ingestion/run_ingest.py) | CLI: `--backfill 2024-01-01:2024-12-31` or `--forecast --forecast-days 14`. Concurrency-bounded via `asyncio.Semaphore`. |
| [ingestion/healthcheck.py](../ingestion/healthcheck.py) | Layered probe (DNS → TCP → HTTP root → HTTP query) for both endpoints. Runs as a preflight before any real ingest. |

### Where the data lands (bronze)

```
s3a://weather-lake/bronze/region=armenia/dataset=forecast/run_ts=20260523T154153Z/lat=41.3000_lon=46.6300.json
s3a://weather-lake/bronze/region=armenia/dataset=archive/year=2024/lat=41.3000_lon=46.6300.json
```

The `key=value` directory layout is **Hive-style partitioning** — it lets Spark figure out partition columns (region, dataset, year, run_ts) automatically without us telling it.

---

## Phase 2a — Spark medallion ETL (cleaning + features)

**Goal:** transform the raw JSON into a clean, model-ready table.

### Why Spark?

For 288 rows of forecast data, pandas would be fine. But this is meant to scale to several years × 100 grid points × hourly cadence = millions of rows. The same code that handles 288 rows handles 28 million if the data shape grows.

Spark also gives us:
- **Lazy evaluation** — chained transformations don't execute until `.write` or `.count`, so the optimizer can fuse them.
- **Partitioned write** — `df.write.partitionBy("date").parquet(...)` produces one Parquet file per date, which lets later queries skip dates they don't need.
- **Standard window functions** for lag / rolling features that would be a pain in plain Python.

### Bronze → Silver: parse and explode

The bronze JSON has **parallel arrays** — `hourly.time[i]` corresponds to `hourly.temperature_2m[i]`. PyTorch wants one row per timestamp, not 10 parallel arrays. So [spark_jobs/bronze_to_silver.py](../spark_jobs/bronze_to_silver.py) does:

```python
zipped = F.arrays_zip(F.col("hourly.time"), *[F.col(f"hourly.{v}") for v in HOURLY_VARIABLES])
exploded = df.select(..., F.explode(zipped).alias("h"))
```

`arrays_zip` + `explode` is the Spark idiom for "interleave parallel arrays and flatten into rows." After this we have one row per `(lat, lon, observed_at)` with 10 weather columns.

We also `dropDuplicates(["region", "dataset", "lat", "lon", "observed_at"])` because forecast files overlap as newer runs arrive (a 14-day forecast pulled today shares hours with the one pulled yesterday).

Result: `s3a://weather-lake/silver/weather_observations/dataset=forecast/date=2026-05-23/part-*.parquet`

### Silver → Gold: feature engineering

[spark_jobs/silver_to_gold.py](../spark_jobs/silver_to_gold.py) adds the kinds of features a sequence model needs:

| Feature family | Examples | Why |
|---|---|---|
| **Time** | `hour`, `day_of_week`, `month`, `season`, `is_weekend` | Cyclical patterns the model can learn (diurnal, seasonal). |
| **Lag** | `temperature_2m_lag_1h`, `_lag_3h`, `_lag_24h` | Recent values are the strongest predictors of next values. |
| **Rolling** | `temperature_2m_roll24_mean`, `_roll24_std` | Local trend and volatility, computed over the previous 24 hourly samples. |

Implemented with Spark window functions, partitioned per `(lat, lon)` so the lag of London doesn't bleed into Tokyo:

```python
by_location = Window.partitionBy("lat", "lon").orderBy("observed_at")
df.withColumn("temperature_2m_lag_24h", F.lag("temperature_2m", 24).over(by_location))
```

Result: `s3a://weather-lake/gold/weather_features/` — 27 columns, **this is what the PyTorch `Dataset` will read in Phase 4.**

### Infra fixes baked into this phase

These aren't interesting in themselves but matter for reproducibility, so they're in the docker-compose:

- **`spark_init`** one-shot container: fixes ownership on the named volume that caches Spark's downloaded JARs. Docker named volumes default to `root:root`, but Spark runs as UID 185 and otherwise can't write to its own cache.
- Removed the **cloud-committer config** from `spark-defaults.conf`. The "directory" committer + `PathOutputCommitProtocol` need extra JARs we don't ship and silently produce `_SUCCESS` with zero data files when broken. Default `FileOutputCommitter` works fine on MinIO.

---

## Phase 2b — Hive Metastore (SQL access to the lake)

**Goal:** make the silver and gold Parquet queryable by name, in standard SQL, from any tool.

### What's the point if Spark can already read Parquet?

`spark.read.parquet("s3a://...")` works fine for Python code that knows the path. But:

- A BI tool like DBeaver / Tableau / Metabase doesn't speak "Parquet path."
- Even from `spark-sql`, you'd have to write the full path every time and re-discover the schema.
- A future ML engineer reading the project would have to know that `s3a://weather-lake/gold/weather_features/` even exists.

The Hive Metastore solves all three. It's a tiny Java service whose only job is to remember:

> *"There's a table called `gold.weather_features` at `s3a://weather-lake/gold/weather_features/`. It has these 27 columns. It's partitioned by `dataset` and `date`. The data is Parquet."*

Once Spark is told `spark.hadoop.hive.metastore.uris = thrift://hive-metastore:9083`, you can do:

```sql
SELECT dataset, date, COUNT(*) FROM gold.weather_features GROUP BY dataset, date;
```

and Spark figures out where to read from.

### The architecture

```
spark-master ─────┐                            ┌── postgres (metastore_db)
                  │                            │   ↑ schema for Hive's metadata
                  ├─► hive-metastore:9083 ─────┤
                  │   (thrift)                 │
spark-worker ─────┘                            └── s3a://weather-lake/...
                                                   ↑ actual data files
```

`hive-metastore` stores its metadata in a separate Postgres database (`metastore_db`) — same Postgres container, different DB. The actual *data* still lives on MinIO; only schema/table info goes in Postgres.

### DDL ([warehouse/hive/create_external_tables.sql](../warehouse/hive/create_external_tables.sql))

```sql
CREATE DATABASE IF NOT EXISTS silver;
CREATE DATABASE IF NOT EXISTS gold;

CREATE TABLE silver.weather_observations USING PARQUET
  LOCATION 's3a://weather-lake/silver/weather_observations';
CREATE TABLE gold.weather_features USING PARQUET
  LOCATION 's3a://weather-lake/gold/weather_features';

MSCK REPAIR TABLE silver.weather_observations;
MSCK REPAIR TABLE gold.weather_features;
```

The `MSCK REPAIR` is unusual: it tells the metastore to scan the LOCATION and register the partition subdirectories (`dataset=forecast/date=2026-05-23/`, etc.) it finds. Without it, Spark trusts the metastore's empty partition list and queries return zero rows even though Parquet exists.

### Why this phase took a while

The compose file ended up with a small zoo of fixes for issues nobody talks about until you hit them:

| Issue | Why it happens | Fix |
|---|---|---|
| `ClassNotFoundException: org.postgresql.Driver` in schematool | Hive 4.0 doesn't auto-download the postgres JDBC driver despite `DB_DRIVER=postgres` | Bind-mount `postgresql-42.7.3.jar` into `/opt/hive/lib/` |
| `ClassNotFoundException: S3AFileSystem` in metastore | The `hadoop-aws` JAR is in `tools/lib` but not on the default classpath | `HADOOP_OPTIONAL_TOOLS=hadoop-aws` env var |
| `NoAuthWithAWSException` from metastore on `s3a://` | `-Dfs.s3a.*` JVM properties don't configure Hadoop (it reads XML, not system props) | Bind-mount [docker/hive/conf/core-site.xml](../docker/hive/conf/core-site.xml) into `/opt/hadoop/etc/hadoop/` |
| `Failed to create external path s3a://.../warehouse/silver.db` | Spark sends a default LOCATION on `CREATE DATABASE`; metastore tries to `mkdir` it | Set `spark.sql.warehouse.dir = file:/tmp/spark-warehouse` so the default goes local; our tables have explicit `s3a://` LOCATIONs |

The reward: real SQL access to the lake.

---

## Phase 3 — Postgres star schema (analytics warehouse)

**Goal:** give human analysts a clean relational schema for ad-hoc queries and BI tools.

### Why a separate Postgres if Hive already gives SQL?

| | Hive-cataloged lake | Postgres star schema |
|---|---|---|
| Storage | Parquet on MinIO | Postgres tables |
| Optimized for | Bulk scans, time-series ML | `GROUP BY`-heavy analytics, indexed joins, BI dashboards |
| Cost of joins | Expensive (no indexes) | Cheap (B-tree indexes) |
| Schema | Wide (27 cols, includes feature columns) | Normalized into fact + dims |
| Mutation | Append-only, rewrite-partition | Real SQL: `UPDATE`, `INSERT`, `DELETE`, transactions |
| Who queries it | Spark, PyTorch | psql, DBeaver, Metabase, dashboards |

The lake is the "training plane." The warehouse is the "human plane." Real companies run both for the same reason: they answer different questions well.

### The star schema

```
                  ┌──────────────────────────┐
                  │ fact_weather_observations│
                  │  • location_id  (FK)     │
                  │  • time_id      (FK)     │
                  │  • dataset      ('forecast'|'archive')
                  │  • temperature_2m        │
                  │  • wind_speed_10m        │
                  │  • ... (9 more measures) │
                  └────┬──────────────┬──────┘
                       │              │
        ┌──────────────▼┐            ┌▼──────────────┐
        │ dim_location  │            │ dim_time      │
        │  location_id  │            │  time_id      │
        │  region       │            │  observed_at  │
        │  lat, lon     │            │  date, hour   │
        │  elevation    │            │  day_of_week  │
        └───────────────┘            │  month, year  │
                                     │  season       │
                                     │  is_weekend   │
                                     └───────────────┘
```

If you're new to relational design:

- **Fact table** = one row per **measurement event**. Mostly numbers (the things you measured) plus foreign-key IDs that point to dimensions.
- **Dimension tables** = lookup tables for the "context" of each measurement (what location was it, what time, etc.). One row per unique location / unique timestamp.
- **Foreign key (FK)** = a column that references the primary key of another table. The database enforces referential integrity — you can't insert a fact row whose `location_id` doesn't exist in `dim_location`.

The win is that the fact table is **narrow and append-only**, while context lives in small dim tables. Queries like "average temperature by month in northern Armenia" become fast index-driven joins.

### The loader ([spark_jobs/load_to_warehouse.py](../spark_jobs/load_to_warehouse.py))

```
silver Parquet
   │
   ├─► distinct (region, lat, lon, elevation) ── row_number() ─► dim_location
   │
   ├─► distinct observed_at ────────────────── row_number() ─► dim_time
   │
   └─► JOIN both back in ───────────────────── (loc_id, time_id, measures) ─► fact
```

Three details worth explaining:

1. **Surrogate keys via `row_number()`**: instead of relying on the database's auto-increment (`SERIAL`), Spark assigns integer IDs by sorting deterministically. This makes the loader fully reproducible — same source → same IDs.

2. **Full-refresh load pattern**: each run starts with `TRUNCATE fact_weather_observations, dim_location, dim_time` in a single statement (Postgres allows truncating multiple FK-related tables in one statement without `CASCADE`), then `mode="append"` for each Spark write. This preserves the schema, indexes, and FK constraints from the DDL — `mode="overwrite"` alone would drop and recreate the tables, losing all that.

3. **The `jdbc_execute` helper**: Spark's writer only does table-level overwrite/append. For raw SQL (`TRUNCATE`), we need to talk JDBC directly from Spark's driver process. Two gotchas hit along the way:
   - `--packages` JARs aren't on the **driver** classpath, only executors — fixed with `spark.driver.extraClassPath` in [spark-defaults.conf](../docker/spark/conf/spark-defaults.conf).
   - `DriverManager` doesn't auto-register Ivy-loaded drivers — fixed with explicit `Class.forName("org.postgresql.Driver")` before `getConnection`.

### What you can do with it now

```bash
docker exec weather_postgres psql -U weather -d weather_dw

> SELECT l.lat, l.lon, MAX(f.temperature_2m) AS hottest
  FROM fact_weather_observations f
  JOIN dim_location l USING (location_id)
  GROUP BY l.lat, l.lon
  ORDER BY hottest DESC;
```

Returns a leaderboard of grid cells by max recorded temperature. The east-southern corner of Armenia is the hottest (22.9 °C in our 3-day forecast window), which matches real geography.

---

## Where the model fits next (Phase 4)

The gold table at `s3a://weather-lake/gold/weather_features/` is **the model's training set**. Phase 4 builds:

1. **`ml/dataset.py`** — a PyTorch `Dataset` that:
   - Reads gold Parquet (probably via pandas or `pyarrow` for simplicity; we don't need Spark in the train loop).
   - Groups rows by `(lat, lon)` (each location is its own time series).
   - Windows each series into `(seq_in: shape [L_in, n_features], seq_out: shape [L_out])` pairs.
   - Holds out the most recent slice for validation (**time-based split — no leakage**).

2. **`ml/models/lstm.py`** — a multivariate LSTM. Input is `[B, L_in, n_features]`, output is `[B, L_out]` (the next L_out temperatures).

3. **`ml/train.py`** — Adam, MSE loss, checkpointing the best validation loss to disk.

4. **`ml/evaluate.py`** — backtest, report MAE / RMSE / MAPE.

### Honest caveat

Right now we have **forecast** data only in bronze (the archive endpoint is down upstream). That means we'd be training the model on Open-Meteo's NWP-derived forecasts — which is "predicting forecasts," not really useful. The code path is identical for historical archive data; once the upstream recovers and we backfill, the same training pipeline produces a real model.

---

## Running the whole thing

For an AI engineer who wants to poke at it:

```powershell
# 1. Bring up infra (MinIO + Postgres + Spark + Hive Metastore)
docker compose up -d

# 2. Activate venv (Phase 1 ingestion runs natively)
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Ingest some data (forecast endpoint is healthy)
python -m ingestion.run_ingest --forecast --forecast-days 14

# 4. Run the Spark ETL chain (inside container)
docker exec weather_spark_master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 /opt/jobs/bronze_to_silver.py
docker exec weather_spark_master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 /opt/jobs/silver_to_gold.py

# 5. (One time) register external tables in Hive Metastore
docker cp warehouse/hive/create_external_tables.sql weather_spark_master:/tmp/
docker exec weather_spark_master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 -f /tmp/create_external_tables.sql

# 6. Apply warehouse DDL and load star schema
Get-Content warehouse/ddl/*.sql | docker exec -i weather_postgres \
  psql -U weather -d weather_dw
docker exec weather_spark_master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 /opt/jobs/load_to_warehouse.py

# 7. Query the lake or the warehouse
docker exec weather_postgres psql -U weather -d weather_dw \
  -c "SELECT COUNT(*) FROM fact_weather_observations;"
```

UIs to poke at:
- **MinIO console**: http://localhost:9001 (`minioadmin` / `minioadmin`) — browse the lake.
- **Spark master UI**: http://localhost:8080 — see running and completed jobs.
- **Spark application UI**: http://localhost:4040 — DAG, stages, executors (only up while a job runs).

---

*This document covers Phases 0–3. Phase 4 (training) will add ML-specific sections when it lands.*
