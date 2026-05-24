"""Data — pick a date range, trigger an additive backfill via the Airflow DAG."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from dashboard.airflow_api import trigger_dag
from dashboard.state import fetch_coverage, reset_caches, sidebar_config

cfg = sidebar_config()
st.title("Data")
st.caption(
    "Extend the gold date range. The Airflow DAG is additive — it only fetches "
    "days that are missing from the manifest, so re-running a covered range is "
    "a no-op."
)

cov = fetch_coverage()
if cov and cov.archive:
    st.info(
        f"Current archive coverage: **{min(cov.archive)} → {max(cov.archive)}** "
        f"({len(cov.archive)} days)."
    )
else:
    st.info("No archive coverage yet.")

today = date.today()
default_start = today - timedelta(days=14)
default_end = today - timedelta(days=2)  # archive lags ~2 days

with st.form("backfill_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        start_d = st.date_input("Start date", value=default_start)
    with col2:
        end_d = st.date_input("End date", value=default_end)
    with col3:
        force = st.checkbox(
            "Force re-fetch",
            value=False,
            help="Ignore the coverage manifest and re-pull every day in the range.",
        )
    submitted = st.form_submit_button("Backfill missing days", type="primary")

if submitted:
    if start_d > end_d:
        st.error("Start date must be ≤ end date.")
    else:
        try:
            run = trigger_dag(
                cfg["dag_id"],
                conf={
                    "start_date": start_d.isoformat(),
                    "end_date": end_d.isoformat(),
                    "force": bool(force),
                },
            )
            st.success(
                f"Triggered `{cfg['dag_id']}` · run_id "
                f"`{run.get('dag_run_id')}` · state `{run.get('state')}`."
            )
            st.caption(
                "Watch progress on the **Status** page or the Airflow UI "
                "(http://localhost:8081)."
            )
            reset_caches()
        except Exception as exc:
            st.error(f"Could not trigger DAG: {exc}")

st.divider()
st.subheader("Manifest")
if cov:
    st.caption(
        f"Manifest scanned at "
        f"`{cov.scanned_at.isoformat() if cov.scanned_at else 'n/a'}`. "
        f"Archive: {len(cov.archive)} dates · Forecast: {len(cov.forecast)} dates."
    )
    with st.expander("Inspect dates"):
        st.write({"archive": sorted(d.isoformat() for d in cov.archive)})
