-- Model registry: one row per training run. The Streamlit dashboard reads
-- this to render "current model: trained on YYYY-MM-DD .. YYYY-MM-DD,
-- gold has data through YYYY-MM-DD (stale!)" badges.
--
-- model_version is an ISO-8601 timestamp set at the start of training, also
-- used as the versioned checkpoint filename (checkpoints/{version}.pt).
-- checkpoints/best.pt is a copy of whichever versioned file has the lowest
-- val MSE; cross-reference is_best below.

CREATE TABLE IF NOT EXISTS models (
    model_version       TEXT               PRIMARY KEY,
    trained_at          TIMESTAMPTZ        NOT NULL,
    data_range_start    DATE,
    data_range_end      DATE,
    gold_row_count      BIGINT,
    best_val_mse        DOUBLE PRECISION,
    epochs              INTEGER,
    is_best             BOOLEAN            NOT NULL DEFAULT FALSE,
    checkpoint_path     TEXT,
    hyperparams         JSONB,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_models_trained_at ON models (trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_is_best ON models (is_best) WHERE is_best;
