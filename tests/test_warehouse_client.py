"""Unit tests for the Postgres client. psycopg2 is mocked end-to-end."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from warehouse.client import (
    PredictionRow,
    insert_predictions,
    transaction,
)


def _mock_conn():
    """Return (conn, cur) where the cursor doubles as a context manager."""
    cur = MagicMock()
    cur.__enter__ = lambda self: self
    cur.__exit__ = lambda self, *a: None
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


def test_insert_predictions_executes_bulk_insert():
    conn, cur = _mock_conn()
    rows = [
        PredictionRow(
            target_time=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
            predicted_value=12.5,
        ),
        PredictionRow(
            target_time=datetime(2026, 5, 24, 13, 0, tzinfo=timezone.utc),
            predicted_value=13.0,
        ),
    ]
    cur.rowcount = 2
    n = insert_predictions(
        conn,
        model_version="2026-05-24T10-00-00Z",
        location_id=42,
        prediction_made_at=datetime(2026, 5, 24, 11, 0, tzinfo=timezone.utc),
        rows=rows,
        seq_in=240,
    )
    assert n == 2
    cur.executemany.assert_called_once()
    sql, params = cur.executemany.call_args.args
    assert "INSERT INTO predictions" in sql
    assert "seq_in" in sql
    assert "ON CONFLICT" in sql
    assert len(params) == 2
    assert params[0] == (
        "2026-05-24T10-00-00Z",
        42,
        datetime(2026, 5, 24, 11, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        12.5,
        240,
    )


def test_insert_predictions_seq_in_defaults_to_null():
    conn, cur = _mock_conn()
    rows = [
        PredictionRow(
            target_time=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
            predicted_value=12.5,
        ),
    ]
    cur.rowcount = 1
    insert_predictions(
        conn,
        model_version="v",
        location_id=1,
        prediction_made_at=datetime(2026, 5, 24, 11, 0, tzinfo=timezone.utc),
        rows=rows,
    )
    _, params = cur.executemany.call_args.args
    assert params[0][-1] is None


def test_insert_predictions_empty_list_short_circuits():
    conn, cur = _mock_conn()
    n = insert_predictions(
        conn,
        model_version="v",
        location_id=1,
        prediction_made_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        rows=[],
    )
    assert n == 0
    cur.executemany.assert_not_called()


def test_transaction_commits_on_clean_exit():
    conn = MagicMock()
    with transaction(conn):
        pass
    conn.commit.assert_called_once()
    conn.rollback.assert_not_called()


def test_transaction_rolls_back_on_exception():
    conn = MagicMock()
    try:
        with transaction(conn):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()
