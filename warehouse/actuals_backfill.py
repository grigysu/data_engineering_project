"""CLI: fill predictions.actual_value from fact_weather_observations.

Run by Airflow after `load_warehouse`, but also safe to run by hand:

    python -m warehouse.actuals_backfill

Idempotent — only updates rows where actual_value IS NULL.
"""

from __future__ import annotations

from dotenv import load_dotenv

from warehouse.client import backfill_actuals, connect_from_env, transaction

load_dotenv()


def main() -> None:
    conn = connect_from_env()
    try:
        with transaction(conn):
            n = backfill_actuals(conn)
        print(f"backfill_actuals: filled {n} prediction rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
