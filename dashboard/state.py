"""Cached resources for the Streamlit dashboard.

Streamlit reruns the script top-to-bottom on every interaction. Without
caching, every keystroke would reload the checkpoint metadata and re-fetch
locations. `@st.cache_data` (read-only DB queries + checkpoint metadata)
solves that.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from warehouse.client import connect_from_env


DEFAULT_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoints/best.pt")
DEFAULT_GOLD = os.getenv("GOLD_PATH", "s3://weather-lake/gold/weather_features")


@st.cache_data(ttl=60, show_spinner=False)
def read_checkpoint_meta(path: str) -> dict:
    """Load only the metadata fields from a checkpoint (skip weights into memory).

    Returns an empty dict if the file is missing — the dashboard renders a
    friendly placeholder instead of crashing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "model_version": ckpt.get("model_version"),
        "trained_at": ckpt.get("trained_at"),
        "data_range_start": ckpt.get("data_range_start"),
        "data_range_end": ckpt.get("data_range_end"),
        "best_val_mse": ckpt.get("best_val_mse"),
        "gold_row_count": ckpt.get("gold_row_count"),
    }


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
