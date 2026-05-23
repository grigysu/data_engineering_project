-- Star-schema fact: one row per (location, time, dataset). Measures are
-- the raw weather variables; lag / rolling features stay in the gold lake
-- (model-facing, not analytics-facing).

DROP TABLE IF EXISTS fact_weather_observations CASCADE;

CREATE TABLE fact_weather_observations (
    location_id           INTEGER          NOT NULL REFERENCES dim_location (location_id),
    time_id               INTEGER          NOT NULL REFERENCES dim_time     (time_id),
    dataset               TEXT             NOT NULL,  -- 'archive' | 'forecast'
    temperature_2m        DOUBLE PRECISION,
    relative_humidity_2m  DOUBLE PRECISION,
    dew_point_2m          DOUBLE PRECISION,
    precipitation         DOUBLE PRECISION,
    pressure_msl          DOUBLE PRECISION,
    cloud_cover           DOUBLE PRECISION,
    wind_speed_10m        DOUBLE PRECISION,
    wind_direction_10m    DOUBLE PRECISION,
    wind_gusts_10m        DOUBLE PRECISION,
    shortwave_radiation   DOUBLE PRECISION,
    PRIMARY KEY (location_id, time_id, dataset)
);

CREATE INDEX idx_fact_time ON fact_weather_observations (time_id);
CREATE INDEX idx_fact_location ON fact_weather_observations (location_id);
CREATE INDEX idx_fact_dataset ON fact_weather_observations (dataset);
