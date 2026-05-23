"""Silver -> Postgres star schema (dim_location, dim_time, fact_weather_observations).

Reads the silver Parquet from MinIO, builds dimension tables with deterministic
surrogate keys (row_number over a stable sort), joins to produce the fact rows,
and writes everything via JDBC to weather_dw.

Loading strategy:
  - DDL (run separately, see warehouse/ddl/) defines the star schema with real
    foreign keys on the fact table — good schema hygiene + showcase value.
  - This job runs in full-refresh mode: TRUNCATE all three tables in a single
    statement (so we don't need CASCADE or to drop FKs), then mode=append the
    Spark DataFrames. Atomic-ish, preserves indexes + constraints.

Run from inside the spark-master container:
    docker exec weather_spark_master \\
        /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        /opt/jobs/load_to_warehouse.py
"""

from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def _season_expr(month_col: F.Column) -> F.Column:
    return (
        F.when(month_col.isin(12, 1, 2), F.lit("winter"))
        .when(month_col.isin(3, 4, 5), F.lit("spring"))
        .when(month_col.isin(6, 7, 8), F.lit("summer"))
        .otherwise(F.lit("autumn"))
    )


def build_dim_location(silver: DataFrame) -> DataFrame:
    distinct = silver.select("region", "lat", "lon", "elevation").distinct()
    # Stable surrogate key: row_number over a deterministic sort. Reproducible
    # across runs as long as the source data doesn't change.
    return distinct.withColumn(
        "location_id",
        F.row_number().over(Window.orderBy("region", "lat", "lon")),
    ).select("location_id", "region", "lat", "lon", "elevation")


def build_dim_time(silver: DataFrame) -> DataFrame:
    distinct = silver.select("observed_at").distinct()
    with_attrs = (
        distinct.withColumn("date", F.to_date("observed_at"))
        .withColumn("hour", F.hour("observed_at").cast("short"))
        .withColumn("day_of_week", F.dayofweek("observed_at").cast("short"))
        .withColumn("month", F.month("observed_at").cast("short"))
        .withColumn("year", F.year("observed_at").cast("short"))
        .withColumn("season", _season_expr(F.col("month")))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7))
    )
    return with_attrs.withColumn(
        "time_id", F.row_number().over(Window.orderBy("observed_at"))
    ).select(
        "time_id",
        "observed_at",
        "date",
        "hour",
        "day_of_week",
        "month",
        "year",
        "season",
        "is_weekend",
    )


def build_fact(
    silver: DataFrame, dim_location: DataFrame, dim_time: DataFrame
) -> DataFrame:
    # Inner joins are safe: dims are built from silver, so every silver row
    # finds a match.
    return (
        silver.join(
            dim_location.select("location_id", "region", "lat", "lon"),
            on=["region", "lat", "lon"],
            how="inner",
        )
        .join(
            dim_time.select("time_id", "observed_at"),
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

    # Cache silver: the loc/time/fact stages all consume it.
    silver.cache()

    dim_location = build_dim_location(silver).cache()
    dim_time = build_dim_time(silver).cache()
    fact = build_fact(silver, dim_location, dim_time)

    n_loc = dim_location.count()
    n_time = dim_time.count()
    print(
        f"[load_to_warehouse] derived dims: {n_loc} locations, {n_time} times",
        flush=True,
    )

    print(
        "[load_to_warehouse] TRUNCATE fact + dims (single statement, no CASCADE needed)",
        flush=True,
    )
    jdbc_execute(
        spark,
        args.jdbc_url,
        props,
        "TRUNCATE fact_weather_observations, dim_location, dim_time",
    )

    print("[load_to_warehouse] writing dim_location", flush=True)
    dim_location.write.jdbc(
        url=args.jdbc_url, table="dim_location", mode="append", properties=props
    )

    print("[load_to_warehouse] writing dim_time", flush=True)
    dim_time.write.jdbc(
        url=args.jdbc_url, table="dim_time", mode="append", properties=props
    )

    print("[load_to_warehouse] writing fact_weather_observations", flush=True)
    fact.write.jdbc(
        url=args.jdbc_url,
        table="fact_weather_observations",
        mode="append",
        properties=props,
    )

    print("[load_to_warehouse] done.", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
