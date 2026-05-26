"""CLI: refresh `backtest_groups` with per-group MSE for completed backtests.

A "group" = one (location_id, model_version, prediction_made_at) tuple in
the `predictions` table — i.e. all `seq_out` rows for one anchor on one
cell. A group is *complete* when every row has actual_value backfilled.

UPSERT semantics make this safe to re-run; existing rows get refreshed
`mse` + `computed_at` if more rows have since been backfilled into the
same group.

Filters to `model_version LIKE 'backtest:%'` — `ml.backtest` tags its runs
that way, so this table is backtest-only by construction.

Run after a backtest:
    python -m warehouse.backtest_groups
"""

from __future__ import annotations

from dotenv import load_dotenv

from warehouse.client import connect_from_env, transaction

load_dotenv()


UPSERT_SQL = """
INSERT INTO backtest_groups (
    location_id, model_version, prediction_made_at, n_hours, mse
)
SELECT
    location_id,
    model_version,
    prediction_made_at,
    COUNT(*) AS n_hours,
    AVG((predicted_value - actual_value) * (predicted_value - actual_value)) AS mse
FROM predictions
WHERE model_version LIKE 'backtest:%%'
GROUP BY location_id, model_version, prediction_made_at
HAVING COUNT(*) = COUNT(actual_value)
ON CONFLICT (location_id, model_version, prediction_made_at) DO UPDATE
SET n_hours     = EXCLUDED.n_hours,
    mse         = EXCLUDED.mse,
    computed_at = now()
"""


def main() -> None:
    conn = connect_from_env()
    try:
        with transaction(conn):
            with conn.cursor() as cur:
                cur.execute(UPSERT_SQL)
                affected = cur.rowcount
        print(f"backtest_groups: upserted {affected} group(s).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
