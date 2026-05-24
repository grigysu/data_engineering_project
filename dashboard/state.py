"""Cached resources for the Streamlit dashboard.

Streamlit reruns the script top-to-bottom on every interaction. Without
caching, every keystroke would reload the gold parquet and re-init the
predictor (slow). `@st.cache_resource` + `@st.cache_data` solve that.

Resources are keyed by their inputs, so changing the checkpoint path or
the gold URI invalidates the cache. TTLs keep things fresh-ish without
manual cache busts on the dev cycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ingestion.coverage import Coverage, read_manifest, scan_gold_coverage
from warehouse.client import (
    ModelInfo,
    connect_from_env,
    get_best_model,
    list_models,
)


DEFAULT_CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT", "checkpoints/best.pt"))
DEFAULT_GOLD = os.getenv("GOLD_PATH", "s3://weather-lake/gold/weather_features")
DEFAULT_CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
DEFAULT_DAG_ID = os.getenv("WEATHER_DAG_ID", "weather_pipeline")


@st.cache_resource(show_spinner="Loading predictor (checkpoint + gold)…")
def get_predictor(checkpoint_path: str, gold_path: str):
    """Build a Predictor once per Streamlit session."""
    from serving.predictor import Predictor

    return Predictor(checkpoint_path, gold_path, persist=True)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_coverage(bucket: str = "weather-lake") -> Coverage | None:
    """Read the ingest manifest (cheap). Falls back to a fresh scan if missing."""
    from ingestion.coverage import s3_client_from_env

    try:
        s3 = s3_client_from_env()
    except Exception as exc:
        st.error(f"MinIO unreachable: {exc}")
        return None
    cov = read_manifest(s3, bucket)
    if cov is None:
        try:
            cov = scan_gold_coverage(s3, bucket)
        except Exception as exc:
            st.warning(f"Could not scan gold coverage: {exc}")
            return None
    return cov


@st.cache_data(ttl=10, show_spinner=False)
def fetch_models(limit: int = 50) -> list[ModelInfo]:
    """Snapshot the model registry."""
    try:
        conn = connect_from_env()
    except Exception as exc:
        st.error(f"Postgres unreachable: {exc}")
        return []
    try:
        return list_models(conn, limit=limit)
    finally:
        conn.close()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_best_model() -> ModelInfo | None:
    try:
        conn = connect_from_env()
    except Exception:
        return None
    try:
        return get_best_model(conn)
    finally:
        conn.close()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_training_log(checkpoint_dir: str, model_version: str | None) -> pd.DataFrame:
    """Read the per-epoch CSV for a given model version, or 'best' if None.

    Returns an empty DataFrame if the file is missing.
    """
    d = Path(checkpoint_dir)
    candidates: list[Path] = []
    if model_version:
        candidates.append(d / f"{model_version}_log.csv")
    candidates.append(d / "training_log.csv")  # legacy filename pre-2b
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=10, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a read-only SQL query against weather_dw and return a DataFrame."""
    try:
        conn = connect_from_env()
    except Exception as exc:
        st.error(f"Postgres unreachable: {exc}")
        return pd.DataFrame()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def reset_caches() -> None:
    """Force-refresh: call after a successful ingest / train trigger."""
    fetch_coverage.clear()
    fetch_models.clear()
    fetch_best_model.clear()
    fetch_training_log.clear()
    run_query.clear()


def sidebar_config() -> dict[str, Any]:
    """Render the shared sidebar config; return the resolved settings."""
    st.sidebar.markdown("### Weather Pipeline")
    st.sidebar.caption("Phase 2c control plane")
    checkpoint = st.sidebar.text_input("Checkpoint path", value=str(DEFAULT_CHECKPOINT))
    gold = st.sidebar.text_input("Gold parquet URI", value=DEFAULT_GOLD)
    if st.sidebar.button("Refresh caches"):
        reset_caches()
        st.rerun()
    return {
        "checkpoint": checkpoint,
        "gold": gold,
        "checkpoint_dir": str(DEFAULT_CHECKPOINT_DIR),
        "dag_id": DEFAULT_DAG_ID,
    }
