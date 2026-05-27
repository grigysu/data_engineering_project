-- One row per *completed* backtest group: a (location_id, model_version,
-- prediction_made_at) tuple whose `seq_out` rows in `predictions` all have
-- actual_value backfilled. `mse` is the per-group AVG((pred - actual)^2);
-- "best" is just `ORDER BY mse LIMIT N` at query time.
--
-- Populated automatically by the daily DAG's `walk_forward` task (which
-- imports warehouse.backtest_groups.UPSERT_SQL), or by running
-- `python -m warehouse.backtest_groups` manually.
-- Metadata only — to recover the 24 (target_time, predicted, actual) rows,
-- JOIN back to `predictions` on (location_id, model_version, prediction_made_at).

CREATE TABLE IF NOT EXISTS backtest_groups (
    location_id        INTEGER NOT NULL REFERENCES dim_location(location_id),
    model_version      TEXT NOT NULL,
    prediction_made_at TIMESTAMPTZ NOT NULL,
    n_hours            INTEGER NOT NULL,
    mse                DOUBLE PRECISION NOT NULL,
    computed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (location_id, model_version, prediction_made_at)
);

CREATE INDEX IF NOT EXISTS idx_backtest_groups_mse ON backtest_groups (mse);
CREATE INDEX IF NOT EXISTS idx_backtest_groups_loc ON backtest_groups (location_id);
