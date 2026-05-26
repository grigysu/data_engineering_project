"""Weather pipeline — choropleth map + per-marz graph.

The daily Airflow DAG owns ingestion, training, and per-region forecast
generation. This page is read-only: marz polygons (filled, colored by
latest observed temperature) → click a marz → graph of actuals continued
by predicted, for that marz's capital city.

Run locally:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
from pathlib import Path

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


REPO_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = REPO_ROOT / "dashboard" / "data" / "armenia_marzes.geojson"
LOCATIONS_JSON_PATH = REPO_ROOT / "ingestion" / "locations.json"


load_dotenv()
st.set_page_config(page_title="Weather Pipeline", layout="wide")
st.title("Weather forecast — Armenia")

# --- Model provenance ---
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


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """{region_name: {capital, population, admin1, ...}} from locations.json."""
    if not LOCATIONS_JSON_PATH.exists():
        return {}
    payload = json.loads(LOCATIONS_JSON_PATH.read_text(encoding="utf-8"))
    return {entry["region"]: entry for entry in payload}


def _format_snapshot(ts, n_hours: int) -> str:
    """User-friendly label: 'May 26, 09:13 · 24h forecast'."""
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return f"{ts.strftime('%b %d, %H:%M')} · {n_hours}h forecast"


geojson = load_geojson()
meta_by_region = load_metadata()

# --- One row per marz: warehouse coords + latest observation ---
locations = run_query(
    """
    SELECT l.location_id, l.region, l.lat, l.lon,
           latest.t  AS latest_temp,
           latest.ts AS latest_observed_at
    FROM dim_location l
    LEFT JOIN LATERAL (
        SELECT f.temperature_2m AS t, t.observed_at AS ts
        FROM fact_weather_observations f
        JOIN dim_time t ON t.time_id = f.time_id
        WHERE f.location_id = l.location_id AND f.dataset = 'archive'
        ORDER BY t.observed_at DESC
        LIMIT 1
    ) latest ON TRUE
    ORDER BY l.region
    """
)
if locations.empty:
    st.warning(
        "No locations yet — wait for the daily Airflow DAG to populate `dim_location` "
        "(or trigger it from http://localhost:8081)."
    )
    st.stop()

# --- Choropleth map ---
st.markdown("### Click a marz")
fig_map = px.choropleth_mapbox(
    locations,
    geojson=geojson,
    locations="region",
    featureidkey="properties.name",
    color="latest_temp",
    color_continuous_scale="RdYlBu_r",
    range_color=(
        float(locations["latest_temp"].min())
        if locations["latest_temp"].notna().any()
        else 0,
        float(locations["latest_temp"].max())
        if locations["latest_temp"].notna().any()
        else 30,
    ),
    hover_data={"region": True, "latest_temp": ":.1f", "location_id": False},
    labels={"latest_temp": "°C (latest)"},
    zoom=6,
    center={"lat": locations["lat"].mean(), "lon": locations["lon"].mean()},
    opacity=0.65,
    height=440,
)
fig_map.update_layout(
    mapbox_style="open-street-map",
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)
event = st.plotly_chart(
    fig_map,
    on_select="rerun",
    key="marzmap",
    selection_mode=("points",),
    use_container_width=True,
)

# --- Selected marz (None = aggregate average across all regions) ---
# Streamlit's plotly on_select returns point_index = row index into the
# DataFrame passed to choropleth_mapbox.
selected_idx: int | None = None
if event and event.get("selection", {}).get("points"):
    point = event["selection"]["points"][0]
    raw_idx = point.get("point_index")
    if raw_idx is None:
        raw_idx = point.get("point_number")
    if raw_idx is not None and 0 <= raw_idx < len(locations):
        selected_idx = int(raw_idx)

if selected_idx is None:
    selected_id: int | None = None
    selected_region = "All regions"
    st.markdown(f"#### All regions · average across {len(locations)} marzes")
else:
    row = locations.iloc[selected_idx]
    selected_id = int(row["location_id"])
    selected_region = str(row["region"])
    extra = meta_by_region.get(selected_region, {})
    capital = extra.get("capital") or selected_region
    pop = extra.get("population")
    header = f"#### {selected_region} · capital **{capital}**"
    if pop:
        header += f" · pop. {int(pop):,}"
    st.markdown(header)

# --- Date-range sliders ---
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
        value=24,
        help="How many hours of the latest forecast to plot.",
    )

# --- Actuals ---
if selected_id is None:
    # Aggregate: average temperature across all marzes at each timestamp.
    actuals = run_query(
        """
        SELECT t.observed_at, AVG(f.temperature_2m) AS temperature_2m
        FROM fact_weather_observations f
        JOIN dim_time t ON t.time_id = f.time_id
        WHERE f.dataset = 'archive'
          AND t.observed_at >= NOW() - (%s || ' days')::interval
        GROUP BY t.observed_at
        ORDER BY t.observed_at
        """,
        (actuals_days,),
    )
else:
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

# --- Snapshot picker ---
# Each DAG run inserts one new `prediction_made_at` snapshot per cell. Showing
# multiple lets the user compare how the forecast evolved across runs.
if selected_id is None:
    # All cells share the same prediction_made_at values per DAG run.
    snapshots_df = run_query(
        """
        SELECT prediction_made_at, COUNT(DISTINCT target_time) AS n_hours
        FROM predictions
        GROUP BY prediction_made_at
        ORDER BY prediction_made_at DESC
        LIMIT 10
        """,
    )
else:
    snapshots_df = run_query(
        """
        SELECT prediction_made_at, COUNT(DISTINCT target_time) AS n_hours
        FROM predictions
        WHERE location_id = %s
        GROUP BY prediction_made_at
        ORDER BY prediction_made_at DESC
        LIMIT 10
        """,
        (selected_id,),
    )
snapshot_options = (
    snapshots_df["prediction_made_at"].tolist() if not snapshots_df.empty else []
)
snapshot_n_hours: dict = (
    dict(zip(snapshots_df["prediction_made_at"], snapshots_df["n_hours"]))
    if not snapshots_df.empty
    else {}
)
# Newest is always rendered; the picker lists only older snapshots to overlay.
newest_snapshot = snapshot_options[0] if snapshot_options else None
older_options = snapshot_options[1:]
extra_snapshots = st.multiselect(
    "Overlay older snapshots",
    options=older_options,
    default=[],
    format_func=lambda ts: _format_snapshot(ts, int(snapshot_n_hours.get(ts, 0))),
    help="Newest is always shown; pick older snapshots to overlay on top.",
)
selected_snapshots = [newest_snapshot, *extra_snapshots] if newest_snapshot else []

# --- Predictions: rows for every selected snapshot, trimmed to horizon ---
if selected_snapshots:
    if selected_id is None:
        # Aggregate: average predicted_value AND actual_value across all cells
        # per (snapshot, target_time). actual_value averages over the rows that
        # have one backfilled (AVG ignores NULLs) so the hover stays honest.
        preds = run_query(
            """
            SELECT target_time,
                   AVG(predicted_value) AS predicted_value,
                   AVG(actual_value)    AS actual_value,
                   prediction_made_at
            FROM (
                SELECT target_time, predicted_value, actual_value,
                       prediction_made_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY prediction_made_at, location_id
                           ORDER BY target_time
                       ) AS rn
                FROM predictions
                WHERE prediction_made_at = ANY(%s)
            ) ranked
            WHERE rn <= %s
            GROUP BY target_time, prediction_made_at
            ORDER BY prediction_made_at, target_time
            """,
            (selected_snapshots, horizon_hours),
        )
    else:
        preds = run_query(
            """
            SELECT target_time, predicted_value, actual_value, prediction_made_at
            FROM (
                SELECT target_time, predicted_value, actual_value,
                       prediction_made_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY prediction_made_at ORDER BY target_time
                       ) AS rn
                FROM predictions
                WHERE location_id = %s AND prediction_made_at = ANY(%s)
            ) ranked
            WHERE rn <= %s
            ORDER BY prediction_made_at, target_time
            """,
            (selected_id, selected_snapshots, horizon_hours),
        )
else:
    preds = pd.DataFrame(
        columns=["target_time", "predicted_value", "actual_value", "prediction_made_at"]
    )


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Drop tz info so naive datetime64[us] and tz-aware columns compare cleanly."""
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


if not actuals.empty:
    actuals = actuals.copy()
    actuals["observed_at"] = _to_naive_utc(actuals["observed_at"])
if not preds.empty:
    preds = preds.copy()
    preds["target_time"] = _to_naive_utc(preds["target_time"])

# --- Combined chart ---
fig = go.Figure()
if not actuals.empty:
    fig.add_trace(
        go.Scatter(
            x=actuals["observed_at"],
            y=actuals["temperature_2m"],
            name="Actual",
            mode="lines",
            line=dict(color="#1f77b4"),
            hovertemplate=(
                "<b>%{x|%b %d, %H:%M}</b><br>%{y:.2f} °C<extra>Actual</extra>"
            ),
        )
    )
if not preds.empty:
    # Newest first so the freshest snapshot's color is stable across reruns.
    snaps_newest_first = sorted(selected_snapshots, reverse=True)
    palette = px.colors.qualitative.Plotly  # 10 well-separated colors
    for i, snap in enumerate(snaps_newest_first):
        group = preds[preds["prediction_made_at"] == snap]
        if group.empty:
            continue
        # Per-point hover text: include actual + error when backfilled.
        hover_text = []
        for _, r in group.iterrows():
            pred = float(r["predicted_value"])
            actual = r["actual_value"]
            if actual is not None and pd.notna(actual):
                actual_f = float(actual)
                hover_text.append(
                    f"Predicted: {pred:.2f} °C<br>"
                    f"Actual: {actual_f:.2f} °C<br>"
                    f"Error: {pred - actual_f:+.2f} °C"
                )
            else:
                hover_text.append(f"Predicted: {pred:.2f} °C<br>Actual: pending")
        x = list(group["target_time"])
        y = list(group["predicted_value"])
        marker_sizes = [7] * len(x)
        marker_symbols = ["circle"] * len(x)
        # Bridge to the actuals line at the model's input anchor. Exact equality
        # between Postgres TIMESTAMPTZ and pandas datetime64 can silently fail
        # on subtle precision differences, so use "nearest actual <= anchor_ts".
        anchor_ts = pd.Timestamp(group["target_time"].min()) - pd.Timedelta(hours=1)
        if not actuals.empty:
            before = actuals[actuals["observed_at"] <= anchor_ts]
            if not before.empty:
                last = before.iloc[-1]
                x = [last["observed_at"]] + x
                y = [float(last["temperature_2m"])] + y
                hover_text = ["Train end (model input)"] + hover_text
                marker_sizes = [15] + marker_sizes
                marker_symbols = ["star"] + marker_symbols
        color = palette[i % len(palette)]
        snap_label = _format_snapshot(snap, int(snapshot_n_hours.get(snap, len(group))))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=pd.Timestamp(snap).strftime("%b %d, %H:%M"),
                mode="lines+markers",
                line=dict(color=color, dash="dash"),
                marker=dict(size=marker_sizes, symbol=marker_symbols),
                text=hover_text,
                hovertemplate=(
                    "<b>%{x|%b %d, %H:%M}</b><br>"
                    "%{text}"
                    f"<extra>{snap_label}</extra>"
                ),
            )
        )
    if snaps_newest_first:
        latest_label = _format_snapshot(
            snaps_newest_first[0],
            int(snapshot_n_hours.get(snaps_newest_first[0], 0)),
        )
        st.caption(f"{len(snaps_newest_first)} snapshot(s) · latest: {latest_label}")
fig.update_layout(
    xaxis_title="time (UTC)",
    yaxis_title="temperature (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

if actuals.empty and preds.empty:
    scope = "any region" if selected_id is None else selected_region
    st.info(
        f"No observations or predictions for {scope} yet. "
        "Wait for the daily Airflow DAG (or trigger it from http://localhost:8081)."
    )
