-- Persisted model predictions. Each /forecast call inserts seq_out rows
-- (one per forecast hour). `actual_value` is NULL until the corresponding
-- observation lands in fact_weather_observations and the nightly Airflow
-- task `backfill_prediction_actuals` joins them.
--
-- Kept outside the dim/fact full-refresh cycle: load_to_warehouse.py
-- TRUNCATEs dim_location + dim_time + fact_weather_observations, but NOT
-- this table. We want prediction history to survive every warehouse rebuild.

-- `seq_in` records the hours of context fed to the LSTM at inference time.
-- Nullable so legacy rows written before the column existed remain valid.
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL          PRIMARY KEY,
    model_version       TEXT               NOT NULL,
    location_id         INTEGER            NOT NULL REFERENCES dim_location (location_id) ON DELETE RESTRICT,
    prediction_made_at  TIMESTAMPTZ        NOT NULL,
    target_time         TIMESTAMPTZ        NOT NULL,
    predicted_value     DOUBLE PRECISION   NOT NULL,
    actual_value        DOUBLE PRECISION,
    actual_filled_at    TIMESTAMPTZ,
    seq_in              INTEGER,
    UNIQUE (model_version, location_id, prediction_made_at, target_time)
);

-- Idempotent column add for databases created before seq_in existed.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS seq_in INTEGER;

CREATE INDEX IF NOT EXISTS idx_predictions_target_time ON predictions (target_time);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_unfilled
    ON predictions (target_time) WHERE actual_value IS NULL;
