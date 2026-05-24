"""Thin wrapper around the Airflow REST API.

Airflow 2.x exposes basic-auth at /api/v1/. We only need three things:
  - trigger a DAG run with conf={start_date, end_date, force}
  - list recent runs (for the "running…" indicator)
  - fetch the latest run state for a DAG

We use httpx synchronously — Streamlit pages are already blocking.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8081"
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "admin"


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=os.getenv("AIRFLOW_BASE_URL", DEFAULT_BASE_URL),
        auth=(
            os.getenv("AIRFLOW_USER", DEFAULT_USER),
            os.getenv("AIRFLOW_PASSWORD", DEFAULT_PASSWORD),
        ),
        timeout=10.0,
    )


def trigger_dag(dag_id: str, conf: dict[str, Any]) -> dict:
    """Kick off a DAG run. Returns the parsed JSON response."""
    payload = {
        "conf": conf,
        "logical_date": datetime.now(timezone.utc).isoformat(),
    }
    with _client() as c:
        r = c.post(f"/api/v1/dags/{dag_id}/dagRuns", json=payload)
        r.raise_for_status()
        return r.json()


def list_recent_runs(dag_id: str, limit: int = 10) -> list[dict]:
    """Most-recent DAG runs first."""
    with _client() as c:
        r = c.get(
            f"/api/v1/dags/{dag_id}/dagRuns",
            params={"limit": limit, "order_by": "-execution_date"},
        )
        r.raise_for_status()
        return r.json().get("dag_runs", [])


def latest_run_state(dag_id: str) -> str | None:
    """One of 'queued', 'running', 'success', 'failed', or None if no runs yet."""
    runs = list_recent_runs(dag_id, limit=1)
    if not runs:
        return None
    return runs[0].get("state")
