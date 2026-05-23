-- Star-schema dimension: physical location (one row per ERA5 grid cell).
--
-- Surrogate key (location_id) is assigned by the loader (Spark row_number),
-- not by SERIAL, because the loader runs in overwrite mode and needs to
-- reproduce stable IDs across runs from the source data alone.

DROP TABLE IF EXISTS dim_location CASCADE;

CREATE TABLE dim_location (
    location_id  INTEGER          PRIMARY KEY,
    region       TEXT             NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    elevation    DOUBLE PRECISION,
    UNIQUE (region, lat, lon)
);

CREATE INDEX idx_dim_location_region ON dim_location (region);
