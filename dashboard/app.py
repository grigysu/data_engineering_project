"""Weather pipeline — map + graph (single page, read-only).

The Airflow DAG owns ingestion, training, and per-cell forecast
generation. This page just reads what's in Postgres and renders it:
click a grid cell on the map → see hourly actuals continued by the
model's latest forecast.

Run locally:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from dashboard.state import (
    DEFAULT_CHECKPOINT,
    read_checkpoint_meta,
    run_query,
)


load_dotenv()
st.set_page_config(page_title="Weather Pipeline", layout="wide")
st.title("Weather forecast — Armenia")

meta = read_checkpoint_meta(DEFAULT_CHECKPOINT)
if meta:
    parts = [f"Model `{meta.get('model_version') or '—'}`"]
    if meta.get("data_range_start") and meta.get("data_range_end"):
        parts.append(
            f"trained on `{meta['data_range_start']} → {meta['data_range_end']}`"
        )
    if meta.get("best_val_mse") is not None:
        parts.append(f"val MSE {meta['best_val_mse']:.4f}")
    st.caption(" · ".join(parts))
else:
    st.caption(f"No checkpoint at `{DEFAULT_CHECKPOINT}` yet — waiting for the DAG.")

# --- 1. Grid cell map ---
locs = run_query(
    "SELECT location_id, lat, lon, region FROM dim_location ORDER BY location_id"
)
if locs.empty:
    st.warning(
        "No locations yet. Wait for the daily Airflow DAG to populate `dim_location` "
        "(or trigger it manually from http://localhost:8081)."
    )
    st.stop()

st.markdown("### Click a grid cell")
fig_map = px.scatter_mapbox(
    locs,
    lat="lat",
    lon="lon",
    hover_data={"location_id": True, "lat": ":.4f", "lon": ":.4f", "region": False},
    zoom=6,
    height=420,
)
fig_map.update_traces(marker=dict(size=11, color="#d62728"))
fig_map.update_layout(
    mapbox_style="open-street-map",
    mapbox_center=dict(lat=locs["lat"].mean(), lon=locs["lon"].mean()),
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)
event = st.plotly_chart(
    fig_map,
    on_select="rerun",
    key="cellmap",
    use_container_width=True,
    selection_mode=("points",),
)

# --- 2. Selected cell (fallback to first cell) ---
# Streamlit's plotly on_select doesn't reliably surface customdata for
# scatter_mapbox; use point_index (the row index into `locs`) instead.
selected_id = int(locs["location_id"].iloc[0])
if event and event.get("selection", {}).get("points"):
    point = event["selection"]["points"][0]
    point_idx = point.get("point_index")
    if point_idx is None:
        point_idx = point.get("point_number")
    if point_idx is not None and 0 <= point_idx < len(locs):
        selected_id = int(locs.iloc[point_idx]["location_id"])

row = locs[locs["location_id"] == selected_id].iloc[0]
st.markdown(
    f"#### Cell `{selected_id}` — ({row['lat']:.4f}, {row['lon']:.4f}) · {row['region']}"
)

# --- 3. Date-range sliders ---
col1, col2 = st.columns(2)
with col1:
    actuals_days = st.slider(
        "Actuals lookback (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="How far back to plot observed temperatures.",
    )
with col2:
    horizon_hours = st.slider(
        "Prediction horizon (hours)",
        min_value=1,
        max_value=24,
        value=6,
        help="How far into the future to plot the latest forecast.",
    )

# --- 4. Actuals (observed temperatures from the warehouse) ---
actuals = run_query(
    """
    SELECT t.observed_at, f.temperature_2m
    FROM fact_weather_observations f
    JOIN dim_time t ON t.time_id = f.time_id
    WHERE f.location_id = %s
      AND f.dataset = 'archive'
      AND t.observed_at >= NOW() - (%s || ' days')::interval
    ORDER BY t.observed_at
    """,
    (selected_id, actuals_days),
)

# --- 5. Predictions (latest snapshot per target_time, future-facing only) ---
preds = run_query(
    """
    SELECT DISTINCT ON (target_time) target_time, predicted_value
    FROM predictions
    WHERE location_id = %s
      AND target_time >= NOW()
      AND target_time <= NOW() + (%s || ' hours')::interval
    ORDER BY target_time, prediction_made_at DESC
    """,
    (selected_id, horizon_hours),
)

# --- 6. Combined chart ---
fig = go.Figure()
if not actuals.empty:
    fig.add_trace(
        go.Scatter(
            x=actuals["observed_at"],
            y=actuals["temperature_2m"],
            name="Actual",
            mode="lines",
            line=dict(color="#1f77b4"),
        )
    )
if not preds.empty:
    fig.add_trace(
        go.Scatter(
            x=preds["target_time"],
            y=preds["predicted_value"],
            name="Predicted",
            mode="lines+markers",
            line=dict(color="#d62728", dash="dash"),
        )
    )
now_iso = pd.Timestamp.utcnow().isoformat()
fig.add_vline(x=now_iso, line=dict(dash="dot", color="gray"))
# Annotation added separately — passing annotation_text to add_vline triggers
# a plotly bug that averages x/y coordinates and crashes on date axes.
fig.add_annotation(
    x=now_iso,
    y=1.0,
    xref="x",
    yref="paper",
    text="now",
    showarrow=False,
    yanchor="bottom",
    font=dict(color="gray"),
)
fig.update_layout(
    xaxis_title="time (UTC)",
    yaxis_title="temperature (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

if actuals.empty and preds.empty:
    st.info(
        "No data for this cell yet. The next Airflow run will populate it "
        "(or trigger one from http://localhost:8081)."
    )
