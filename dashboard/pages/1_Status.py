"""Status — what's where, when was it touched, is anything stale?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.airflow_api import latest_run_state, list_recent_runs
from dashboard.state import (
    fetch_best_model,
    fetch_coverage,
    fetch_models,
    sidebar_config,
)

cfg = sidebar_config()

st.title("Status")

st.subheader("Ingested coverage")
cov = fetch_coverage()
if cov is None or not cov.archive:
    st.info("No archive coverage. Go to **Data** to start a backfill.")
else:
    st.metric("Archive dates", len(cov.archive))
    st.caption(f"Range: **{min(cov.archive)}** → **{max(cov.archive)}**")
    st.caption(
        f"Manifest scanned at {cov.scanned_at.isoformat() if cov.scanned_at else 'n/a'}"
    )

st.divider()
st.subheader("Model registry")
best = fetch_best_model()
models = fetch_models(limit=20)
if not models:
    st.info("No models registered yet. Train one from the **Train** page.")
else:
    if best:
        st.success(
            f"Best: `{best.model_version}` · val MSE {best.best_val_mse:.4f} · "
            f"trained on `{best.data_range_start} → {best.data_range_end}`"
        )
        if best.data_range_end and cov and cov.archive:
            gold_end = max(cov.archive)
            if gold_end > best.data_range_end:
                st.warning(
                    f"Stale — gold goes to **{gold_end}**, model was trained to "
                    f"**{best.data_range_end}**."
                )
    df = pd.DataFrame(
        [
            {
                "version": m.model_version,
                "trained_at": m.trained_at,
                "data_start": m.data_range_start,
                "data_end": m.data_range_end,
                "rows": m.gold_row_count,
                "epochs": m.epochs,
                "best_val_mse": m.best_val_mse,
                "is_best": m.is_best,
            }
            for m in models
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Airflow")
try:
    state = latest_run_state(cfg["dag_id"])
    st.caption(f"Latest `{cfg['dag_id']}` run state: **{state or 'no runs yet'}**")
    runs = list_recent_runs(cfg["dag_id"], limit=5)
    if runs:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run_id": r.get("dag_run_id"),
                        "state": r.get("state"),
                        "execution_date": r.get("execution_date"),
                        "external_trigger": r.get("external_trigger"),
                    }
                    for r in runs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
except Exception as exc:
    st.warning(f"Airflow API unreachable: {exc}")
