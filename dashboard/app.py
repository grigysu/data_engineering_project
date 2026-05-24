"""Streamlit entry for the weather pipeline control plane.

Run locally:
    streamlit run dashboard/app.py --server.port 8501

The sidebar holds shared config (paths, refresh button); each page in
`dashboard/pages/` is auto-discovered by Streamlit's multipage routing.
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from dashboard.state import (
    DEFAULT_GOLD,
    fetch_best_model,
    fetch_coverage,
    sidebar_config,
)

load_dotenv()

st.set_page_config(
    page_title="Weather Pipeline",
    page_icon=None,
    layout="wide",
)

cfg = sidebar_config()

st.title("Weather Pipeline — control plane")
st.caption(
    "One place to see what data exists, expand the date range, retrain the "
    "model, browse the warehouse, and compare predictions vs. actuals."
)

# Quick-glance summary on the landing page; full detail lives on each page.
col1, col2, col3 = st.columns(3)

cov = fetch_coverage()
with col1:
    st.subheader("Ingested data")
    if cov and cov.archive:
        st.metric(
            "Archive dates",
            len(cov.archive),
            help=f"{min(cov.archive)} → {max(cov.archive)}",
        )
    else:
        st.info("No archive data yet. Use the **Data** page to backfill.")

best = fetch_best_model()
with col2:
    st.subheader("Current model")
    if best:
        st.metric(
            "Best val MSE",
            f"{best.best_val_mse:.4f}" if best.best_val_mse is not None else "—",
        )
        st.caption(
            f"Version `{best.model_version}` · trained on "
            f"`{best.data_range_start} → {best.data_range_end}`"
            if best.data_range_start
            else f"Version `{best.model_version}`"
        )
        # Stale-data warning: if gold extends past the model's trained range.
        if best.data_range_end and cov and cov.archive:
            gold_end = max(cov.archive)
            if gold_end > best.data_range_end:
                st.warning(
                    f"Model is stale — gold has data through **{gold_end}** "
                    f"but model was trained up to **{best.data_range_end}**. "
                    "Retrain from the **Train** page."
                )
    else:
        st.info("No registered models yet. Train one from the **Train** page.")

with col3:
    st.subheader("Gold lake")
    st.caption(f"`{DEFAULT_GOLD}`")
    st.caption(f"Checkpoint: `{cfg['checkpoint']}`")
    st.caption(
        "Use the sidebar pages: **Data** to extend the range, **Train** for the "
        "model, **Predict** to forecast a cell, **Browse** for the warehouse."
    )

st.divider()
st.markdown(
    """
    ### How to use this dashboard

    1. **Data** — pick a date range and click *Backfill missing days*. The
       Airflow DAG runs ingestion → bronze → silver → gold → warehouse.
    2. **Train** — start a fresh training run, or continue an existing one for
       a few more epochs. Watch the loss curves live (Plotly).
    3. **Predict** — pick a grid cell. The model's forecast is written to
       Postgres, and the **Predicted vs. Actual** chart shows historical
       predictions against the observations that arrived later.
    4. **Browse** — quick SQL summaries, or jump out to Adminer for ad-hoc
       queries.
    """
)
