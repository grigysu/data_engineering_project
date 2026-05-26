"""Bronze -> Silver: parse raw Open-Meteo JSON, explode hourly arrays into row-per-timestamp.

Bronze layout (Hive-style paths under s3a://weather-lake/bronze/):
    region={region}/dataset={archive|forecast}/...partitions.../lat=...lon=....json

Each bronze file is one Open-Meteo response with parallel arrays under `hourly`
(time[], temperature_2m[], ...). Silver flattens these to one row per
(lat, lon, observed_at) with typed columns + partition columns we'll need
downstream (date, dataset, region).

Run from inside the spark-master container:
    docker exec weather_spark_master \\
        /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        /opt/jobs/bronze_to_silver.py
"""

from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)


# Must mirror ingestion/schemas.py:HOURLY_VARIABLE_NAMES.
HOURLY_VARIABLES = (
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


def bronze_schema() -> StructType:
    """Explicit schema for bronze JSON files.

    Explicit > inferred: inference is slow on JSON, can pick wrong types when
    a variable is all-null in a file, and breaks silently when Open-Meteo adds
    new fields. The `extra` columns (timezone_abbreviation, generationtime_ms)
    are tolerated by Spark — unknown JSON keys are dropped, not errored on.
    """
    hourly_fields = [StructField("time", ArrayType(StringType()))]
    hourly_fields += [
        StructField(name, ArrayType(DoubleType())) for name in HOURLY_VARIABLES
    ]
    return StructType(
        [
            StructField("latitude", DoubleType()),
            StructField("longitude", DoubleType()),
            StructField("timezone", StringType()),
            StructField("elevation", DoubleType()),
            StructField("utc_offset_seconds", LongType()),
            StructField("hourly_units", MapType(StringType(), StringType())),
            StructField("hourly", StructType(hourly_fields)),
        ]
    )


def read_bronze(spark: SparkSession, bronze_path: str) -> DataFrame:
    """Read bronze JSON with explicit schema + Hive partition discovery.

    The `marz=*` glob picks up the admin-1 partition added by
    `ingestion/run_ingest._marz_partition` so silver carries the marz name
    through to dim_location.
    """
    return (
        spark.read.option("basePath", bronze_path)
        .schema(bronze_schema())
        .json(f"{bronze_path}/region=*/dataset=*/marz=*")
    )


def to_silver(df: DataFrame) -> DataFrame:
    """Flatten parallel hourly arrays to one row per (lat, lon, observed_at)."""
    zipped = F.arrays_zip(
        F.col("hourly.time").alias("time"),
        *[F.col(f"hourly.{v}").alias(v) for v in HOURLY_VARIABLES],
    )
    exploded = df.select(
        F.col("region"),
        F.col("dataset"),
        F.col("marz"),
        F.col("latitude").alias("lat"),
        F.col("longitude").alias("lon"),
        F.col("elevation"),
        F.explode(zipped).alias("h"),
    )
    silver = exploded.select(
        F.col("region"),
        F.col("dataset"),
        F.col("marz"),
        F.col("lat"),
        F.col("lon"),
        F.col("elevation"),
        F.to_timestamp(F.col("h.time")).alias("observed_at"),
        *[F.col(f"h.{v}").alias(v) for v in HOURLY_VARIABLES],
    ).withColumn("date", F.to_date(F.col("observed_at")))

    # Dedupe: forecast files for the same hour overwrite as newer arrives;
    # archive files don't usually overlap but it's cheap insurance.
    return silver.dropDuplicates(
        ["region", "dataset", "marz", "lat", "lon", "observed_at"]
    )


def write_silver(df: DataFrame, silver_path: str) -> None:
    (df.write.mode("overwrite").partitionBy("dataset", "date").parquet(silver_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bronze -> Silver transform.")
    parser.add_argument(
        "--bronze",
        default="s3a://weather-lake/bronze",
        help="Bronze root path. Default: s3a://weather-lake/bronze",
    )
    parser.add_argument(
        "--silver",
        default="s3a://weather-lake/silver/weather_observations",
        help="Silver output path. Default: s3a://weather-lake/silver/weather_observations",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("bronze_to_silver")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[bronze_to_silver] reading from {args.bronze}", flush=True)
    bronze = read_bronze(spark, args.bronze)
    n_bronze = bronze.count()
    if n_bronze == 0:
        print("[bronze_to_silver] no bronze files found — nothing to do.", flush=True)
        spark.stop()
        sys.exit(0)
    print(f"[bronze_to_silver] {n_bronze:,} bronze files", flush=True)

    silver = to_silver(bronze)
    n_rows = silver.count()
    print(
        f"[bronze_to_silver] {n_rows:,} silver rows after explode + dedupe", flush=True
    )

    print(f"[bronze_to_silver] writing to {args.silver}", flush=True)
    write_silver(silver, args.silver)

    print("[bronze_to_silver] done.", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
