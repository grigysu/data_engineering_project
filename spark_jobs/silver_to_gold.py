"""Silver -> Gold: feature engineering for the ML training set.

Adds three families of features keyed by (lat, lon, observed_at):

  * Time features: hour, day_of_week, month, season, is_weekend.
  * Lag features: previous-hour, 3-hour-ago, 24-hour-ago values for the
    target variable (temperature_2m by default) — these are the model's
    direct inputs.
  * Rolling features: 24-hour rolling mean and std for temperature_2m —
    capture local trend / volatility.

The output is one row per (lat, lon, observed_at), keeping the partition
columns (dataset, date) so downstream code can read the same way silver is read.

Run from inside the spark-master container:
    docker exec weather_spark_master \\
        /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        /opt/jobs/silver_to_gold.py
"""

from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


# Season buckets, meteorological convention (DJF / MAM / JJA / SON).
def _season_expr(month_col: F.Column) -> F.Column:
    return (
        F.when(month_col.isin(12, 1, 2), F.lit("winter"))
        .when(month_col.isin(3, 4, 5), F.lit("spring"))
        .when(month_col.isin(6, 7, 8), F.lit("summer"))
        .otherwise(F.lit("autumn"))
    )


def add_time_features(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("hour", F.hour("observed_at"))
        .withColumn("day_of_week", F.dayofweek("observed_at"))  # 1=Sunday
        .withColumn("month", F.month("observed_at"))
        .withColumn("season", _season_expr(F.col("month")))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
    )


def add_lag_and_rolling_features(
    df: DataFrame, target: str = "temperature_2m"
) -> DataFrame:
    """Lag and 24-hour rolling features partitioned per (lat, lon)."""
    by_location = Window.partitionBy("lat", "lon").orderBy("observed_at")

    # Rolling 24-hour window: 23 rows back + current row = 24 hourly samples.
    rolling_24h = (
        Window.partitionBy("lat", "lon")
        .orderBy(F.col("observed_at").cast("long"))
        .rangeBetween(-23 * 3600, 0)
    )

    return (
        df.withColumn(f"{target}_lag_1h", F.lag(target, 1).over(by_location))
        .withColumn(f"{target}_lag_3h", F.lag(target, 3).over(by_location))
        .withColumn(f"{target}_lag_24h", F.lag(target, 24).over(by_location))
        .withColumn(f"{target}_roll24_mean", F.avg(target).over(rolling_24h))
        .withColumn(f"{target}_roll24_std", F.stddev_samp(target).over(rolling_24h))
    )


def build_gold(silver: DataFrame, target: str = "temperature_2m") -> DataFrame:
    return add_lag_and_rolling_features(add_time_features(silver), target=target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver -> Gold feature engineering.")
    parser.add_argument(
        "--silver",
        default="s3a://weather-lake/silver/weather_observations",
        help="Silver input path.",
    )
    parser.add_argument(
        "--gold",
        default="s3a://weather-lake/gold/weather_features",
        help="Gold output path.",
    )
    parser.add_argument(
        "--target",
        default="temperature_2m",
        help="Variable to build lag / rolling features for.",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("silver_to_gold")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[silver_to_gold] reading silver from {args.silver}", flush=True)
    silver = spark.read.parquet(args.silver)
    n_silver = silver.count()
    if n_silver == 0:
        print("[silver_to_gold] no silver rows — nothing to do.", flush=True)
        spark.stop()
        sys.exit(0)
    print(f"[silver_to_gold] {n_silver:,} silver rows", flush=True)

    gold = build_gold(silver, target=args.target)
    print(f"[silver_to_gold] {len(gold.columns)} columns in gold output", flush=True)

    print(f"[silver_to_gold] writing gold to {args.gold}", flush=True)
    gold.write.mode("overwrite").partitionBy("dataset", "date").parquet(args.gold)

    print("[silver_to_gold] done.", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
