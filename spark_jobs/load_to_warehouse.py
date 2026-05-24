"""Silver -> Postgres star schema (dim_location, dim_time, fact_weather_observations).

Reads the silver Parquet from MinIO, upserts dims (so existing surrogate
IDs survive across runs — predictions reference dim_location.location_id),
joins to produce the fact rows, and writes everything via JDBC to weather_dw.

Loading strategy:
  - Dims use Postgres IDENTITY for IDs. The loader writes a stage table per
    dim, then upserts on the natural key (`(region, lat, lon)` and
    `observed_at`) with `ON CONFLICT DO NOTHING`. Existing rows keep their
    IDs; new cells/timestamps get fresh ones.
  - Fact gets TRUNCATE+INSERT (it has no inbound FKs). After the dim upsert,
    we read the persisted dims back through JDBC to pick up IDENTITY-assigned
    IDs, then join silver against them to build the fact rows.

Run from inside the spark-master container:
    docker exec weather_spark_master \\
        /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        /opt/jobs/load_to_warehouse.py
"""

from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _season_expr(month_col: F.Column) -> F.Column:
    return (
        F.when(month_col.isin(12, 1, 2), F.lit("winter"))
        .when(month_col.isin(3, 4, 5), F.lit("spring"))
        .when(month_col.isin(6, 7, 8), F.lit("summer"))
        .otherwise(F.lit("autumn"))
    )


def build_dim_location_stage(silver: DataFrame) -> DataFrame:
    """Distinct cells, no surrogate key — Postgres assigns it on insert."""
    return silver.select("region", "lat", "lon", "elevation").distinct()


def build_dim_time_stage(silver: DataFrame) -> DataFrame:
    """Distinct timestamps + derived attributes — no surrogate key."""
    distinct = silver.select("observed_at").distinct()
    return (
        distinct.withColumn("date", F.to_date("observed_at"))
        .withColumn("hour", F.hour("observed_at").cast("short"))
        .withColumn("day_of_week", F.dayofweek("observed_at").cast("short"))
        .withColumn("month", F.month("observed_at").cast("short"))
        .withColumn("year", F.year("observed_at").cast("short"))
        .withColumn("season", _season_expr(F.col("month")))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7))
    )


def build_fact(
    silver: DataFrame,
    dim_location_persisted: DataFrame,
    dim_time_persisted: DataFrame,
) -> DataFrame:
    """Join silver against the persisted dims to pick up IDENTITY-assigned IDs."""
    return (
        silver.join(
            dim_location_persisted.select("location_id", "region", "lat", "lon"),
            on=["region", "lat", "lon"],
            how="inner",
        )
        .join(
            dim_time_persisted.select("time_id", "observed_at"),
            on=["observed_at"],
            how="inner",
        )
        .select(
            "location_id",
            "time_id",
            "dataset",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "pressure_msl",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
        )
    )


def jdbc_execute(spark: SparkSession, url: str, properties: dict, sql: str) -> None:
    """Run a single SQL statement on the JDBC URL via the Spark driver's JVM.

    `Class.forName` is needed because DriverManager only auto-discovers drivers
    loaded by the system classloader; --packages JARs are loaded by Spark's
    classloader and need explicit registration.
    """
    spark._jvm.java.lang.Class.forName(properties["driver"])
    java_props = spark._jvm.java.util.Properties()
    for k, v in properties.items():
        java_props.setProperty(k, v)
    conn = spark._jvm.java.sql.DriverManager.getConnection(url, java_props)
    try:
        stmt = conn.createStatement()
        try:
            stmt.execute(sql)
        finally:
            stmt.close()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver -> Postgres star schema.")
    parser.add_argument(
        "--silver", default="s3a://weather-lake/silver/weather_observations"
    )
    parser.add_argument(
        "--jdbc-url", default="jdbc:postgresql://postgres:5432/weather_dw"
    )
    parser.add_argument("--user", default="weather")
    parser.add_argument("--password", default="weather")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("load_to_warehouse")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    props = {
        "user": args.user,
        "password": args.password,
        "driver": "org.postgresql.Driver",
    }

    print(f"[load_to_warehouse] reading silver from {args.silver}", flush=True)
    silver = spark.read.parquet(args.silver)
    n_silver = silver.count()
    if n_silver == 0:
        print("[load_to_warehouse] no silver rows — nothing to load.", flush=True)
        spark.stop()
        sys.exit(0)
    print(f"[load_to_warehouse] {n_silver:,} silver rows", flush=True)

    silver.cache()

    dim_location_stage = build_dim_location_stage(silver).cache()
    dim_time_stage = build_dim_time_stage(silver).cache()
    n_loc = dim_location_stage.count()
    n_time = dim_time_stage.count()
    print(
        f"[load_to_warehouse] stage dims: {n_loc} locations, {n_time} times",
        flush=True,
    )

    # 1. Stage dims into temp tables (overwrite — Spark drops + recreates).
    print("[load_to_warehouse] writing dim_location_stage", flush=True)
    dim_location_stage.write.jdbc(
        url=args.jdbc_url,
        table="dim_location_stage",
        mode="overwrite",
        properties=props,
    )
    print("[load_to_warehouse] writing dim_time_stage", flush=True)
    dim_time_stage.write.jdbc(
        url=args.jdbc_url,
        table="dim_time_stage",
        mode="overwrite",
        properties=props,
    )

    # 2. Upsert into the real dims on the natural key (FK from predictions safe).
    print("[load_to_warehouse] upserting dim_location", flush=True)
    jdbc_execute(
        spark,
        args.jdbc_url,
        props,
        """
        INSERT INTO dim_location (region, lat, lon, elevation)
        SELECT region, lat, lon, elevation FROM dim_location_stage
        ON CONFLICT (region, lat, lon) DO NOTHING
        """,
    )
    print("[load_to_warehouse] upserting dim_time", flush=True)
    jdbc_execute(
        spark,
        args.jdbc_url,
        props,
        """
        INSERT INTO dim_time (
            observed_at, date, hour, day_of_week, month, year, season, is_weekend
        )
        SELECT observed_at, date, hour, day_of_week, month, year, season, is_weekend
        FROM dim_time_stage
        ON CONFLICT (observed_at) DO NOTHING
        """,
    )

    # 3. Read the persisted dims back to pick up IDENTITY-assigned IDs.
    dim_location_persisted = spark.read.jdbc(
        url=args.jdbc_url, table="dim_location", properties=props
    )
    dim_time_persisted = spark.read.jdbc(
        url=args.jdbc_url, table="dim_time", properties=props
    )

    # 4. Build + write fact (no inbound FKs → TRUNCATE is safe).
    fact = build_fact(silver, dim_location_persisted, dim_time_persisted)
    print(
        "[load_to_warehouse] TRUNCATE fact_weather_observations (only fact)",
        flush=True,
    )
    jdbc_execute(spark, args.jdbc_url, props, "TRUNCATE fact_weather_observations")

    print("[load_to_warehouse] writing fact_weather_observations", flush=True)
    fact.write.jdbc(
        url=args.jdbc_url,
        table="fact_weather_observations",
        mode="append",
        properties=props,
    )

    # 5. Drop the stage tables.
    print("[load_to_warehouse] dropping stage tables", flush=True)
    jdbc_execute(
        spark,
        args.jdbc_url,
        props,
        "DROP TABLE IF EXISTS dim_location_stage; DROP TABLE IF EXISTS dim_time_stage",
    )

    print("[load_to_warehouse] done.", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
