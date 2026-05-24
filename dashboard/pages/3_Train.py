"""Train — list checkpoints, plot loss curves, kick off (re)training."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.airflow_api import trigger_dag
from dashboard.state import (
    fetch_models,
    fetch_training_log,
    reset_caches,
    sidebar_config,
)

cfg = sidebar_config()
st.title("Train")
st.caption(
    "Each training run records a row in the Postgres `models` table and writes "
    "a versioned checkpoint to disk. `best.pt` always points at the lowest "
    "val-MSE run."
)

models = fetch_models(limit=50)

if not models:
    st.info("No registered models yet — trigger one below.")
else:
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
                "checkpoint": m.checkpoint_path,
            }
            for m in models
        ]
    )
    st.subheader("Registered models")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Loss curves")
    version_options = [m.model_version for m in models]
    pick = st.selectbox("Select a model version", version_options)
    log_df = fetch_training_log(cfg["checkpoint_dir"], pick)
    if log_df.empty:
        st.caption(f"No training log on disk for `{pick}`.")
    else:
        long = log_df.melt(
            id_vars=["epoch"],
            value_vars=["train_mse", "val_mse"],
            var_name="split",
            value_name="mse",
        )
        fig = px.line(
            long,
            x="epoch",
            y="mse",
            color="split",
            markers=True,
            title=f"Loss curves — {pick}",
        )
        fig.update_layout(yaxis_title="MSE", xaxis_title="epoch")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Training actions ---
st.subheader("Trigger training")
st.caption(
    "Training runs inside the Airflow `train_model` task. Use the buttons below "
    "to kick off the same DAG that runs daily."
)

col1, col2 = st.columns(2)
with col1:
    if st.button(
        "Train fresh (full DAG)",
        help="Run the full DAG: ingest_archive → silver → gold → warehouse → train.",
        use_container_width=True,
    ):
        try:
            run = trigger_dag(cfg["dag_id"], conf={})
            st.success(f"Triggered fresh run `{run.get('dag_run_id')}`.")
            reset_caches()
        except Exception as exc:
            st.error(f"Could not trigger DAG: {exc}")

with col2:
    st.caption(
        "Continue-training (`--resume <ckpt> --extra-epochs N`) needs to land in "
        "the Airflow DAG as a separate task before it can be triggered from the "
        "UI. For now, run from a shell: `python -m ml.train --resume ... --extra-epochs N`."
    )
