-- External Hive tables over the silver + gold Parquet on MinIO.
--
-- Run from inside the spark-master container:
--   docker exec -i weather_spark_master /opt/spark/bin/spark-sql \
--     --master spark://spark-master:7077 \
--     -f /opt/jobs/../warehouse/hive/create_external_tables.sql
--
-- Or interactively:
--   docker exec -it weather_spark_master /opt/spark/bin/spark-sql \
--     --master spark://spark-master:7077

-- Databases here are just namespaces — no default LOCATION. Tables have
-- explicit LOCATIONs. (Setting a DB-level s3a:// LOCATION makes the
-- metastore try to mkdir the path, which surfaces s3a config issues.)
CREATE DATABASE IF NOT EXISTS silver
  COMMENT 'Cleaned, typed, deduped weather observations.';

CREATE DATABASE IF NOT EXISTS gold
  COMMENT 'Feature-engineered training set for the ML pipeline.';

-- Let Spark infer schema + partition columns from the Parquet path layout.
-- (`USING PARQUET` with no explicit column list also can't take a separate
-- PARTITIONED BY clause; Spark figures it out from the files.)

DROP TABLE IF EXISTS silver.weather_observations;
CREATE TABLE silver.weather_observations
USING PARQUET
LOCATION 's3a://weather-lake/silver/weather_observations';

DROP TABLE IF EXISTS gold.weather_features;
CREATE TABLE gold.weather_features
USING PARQUET
LOCATION 's3a://weather-lake/gold/weather_features';

-- Register existing partition directories with the metastore. Without this,
-- queries against the tables return zero rows because `Partition Provider:
-- Catalog` means Spark trusts the metastore's partition list — and CREATE
-- TABLE doesn't auto-scan partition dirs.
MSCK REPAIR TABLE silver.weather_observations;
MSCK REPAIR TABLE gold.weather_features;
