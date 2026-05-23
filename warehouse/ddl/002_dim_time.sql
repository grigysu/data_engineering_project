-- Star-schema dimension: time (one row per distinct observation timestamp).
--
-- We don't pre-populate a full hourly calendar — only times that appear in
-- the fact get a row, regenerated on each load. Cheap to expand later if
-- we want a true date-spine for left-joining missing hours.

DROP TABLE IF EXISTS dim_time CASCADE;

CREATE TABLE dim_time (
    time_id      INTEGER     PRIMARY KEY,
    observed_at  TIMESTAMP   NOT NULL UNIQUE,
    date         DATE        NOT NULL,
    hour         SMALLINT    NOT NULL,
    day_of_week  SMALLINT    NOT NULL,   -- 1=Sunday, matches Spark dayofweek
    month        SMALLINT    NOT NULL,
    year         SMALLINT    NOT NULL,
    season       TEXT        NOT NULL,   -- 'winter' | 'spring' | 'summer' | 'autumn'
    is_weekend   BOOLEAN     NOT NULL
);

CREATE INDEX idx_dim_time_date ON dim_time (date);
CREATE INDEX idx_dim_time_year_month ON dim_time (year, month);
