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

# --- Actuals (full history) ---
if selected_id is None:
    # Aggregate: average temperature across all marzes at each timestamp.
    actuals = run_query(
        """
        SELECT t.observed_at, AVG(f.temperature_2m) AS temperature_2m
        FROM fact_weather_observations f
        JOIN dim_time t ON t.time_id = f.time_id
        WHERE f.dataset = 'archive'
        GROUP BY t.observed_at
        ORDER BY t.observed_at
        """,
    )
else:
    actuals = run_query(
        """
        SELECT t.observed_at, f.temperature_2m
        FROM fact_weather_observations f
        JOIN dim_time t ON t.time_id = f.time_id
        WHERE f.location_id = %s
          AND f.dataset = 'archive'
        ORDER BY t.observed_at
        """,
        (selected_id,),
    )

# --- Unified selection state ---
# A "selection key" encodes either a DAG snapshot (`dag|<iso>`) or a backtest
# group (`bt|<loc_id>|<model_version>|<iso>`). Both live in one combobox so
# checking a row in the backtest table puts it on the combobox's chip list.


def _encode_dag(pma) -> str:
    return f"dag|{pd.Timestamp(pma).isoformat()}"


def _encode_bt(loc_id: int, mv: str, pma) -> str:
    return f"bt|{loc_id}|{mv}|{pd.Timestamp(pma).isoformat()}"


def _is_bt(key: str) -> bool:
    return key.startswith("bt|")


def _decode_dag_pma(key: str):
    return pd.Timestamp(key.split("|", 1)[1])


# 1. DAG snapshot options — walk_forward is the only writer of predictions now.
if selected_id is None:
    snapshots_df = run_query(
        """
        SELECT prediction_made_at, COUNT(DISTINCT target_time) AS n_hours
        FROM predictions
        GROUP BY prediction_made_at
        ORDER BY prediction_made_at DESC
        LIMIT 50
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
        LIMIT 50
        """,
        (selected_id,),
    )
# The newest DAG snapshot is always on the chart and is NOT a combobox option
# (user can't deselect it). Older snapshots are available in the combobox.
all_dag_pmas = (
    list(snapshots_df["prediction_made_at"]) if not snapshots_df.empty else []
)
newest_dag_pma = all_dag_pmas[0] if all_dag_pmas else None
selectable_dag_pmas = all_dag_pmas[1:] if all_dag_pmas else []
dag_options = [_encode_dag(s) for s in selectable_dag_pmas]
# Keep n_hours for ALL DAG snapshots (including newest) so the label lookup
# still works in the trace loop below.
dag_n_hours: dict = (
    {
        _encode_dag(row["prediction_made_at"]): int(row["n_hours"])
        for _, row in snapshots_df.iterrows()
    }
    if not snapshots_df.empty
    else {}
)

# 2. Backtest table — rendered FIRST so we know what's checked before the combobox.
bt_header = "#### Best backtest groups (lowest MSE first)"
if selected_id is not None:
    bt_header += f" — {selected_region} only"
st.markdown(bt_header)

_BT_SQL = """
    WITH best_per_anchor AS (
        SELECT DISTINCT ON (b.location_id, b.prediction_made_at)
               b.location_id, b.model_version, b.prediction_made_at,
               b.n_hours, b.mse
        FROM backtest_groups b
        {where}
        ORDER BY b.location_id, b.prediction_made_at, b.mse ASC
    )
    SELECT bpa.location_id, dl.region, bpa.model_version,
           bpa.prediction_made_at, bpa.n_hours, bpa.mse,
           -- All `predictions` rows in a single insert_predictions() batch
           -- share the same seq_in, so MAX collapses them safely (and is
           -- NULL-tolerant for legacy rows written before the column existed).
           MAX(p.seq_in) AS seq_in
    FROM best_per_anchor bpa
    JOIN dim_location dl ON dl.location_id = bpa.location_id
    LEFT JOIN predictions p
        ON p.location_id = bpa.location_id
       AND p.model_version = bpa.model_version
       AND p.prediction_made_at = bpa.prediction_made_at
    GROUP BY bpa.location_id, dl.region, bpa.model_version,
             bpa.prediction_made_at, bpa.n_hours, bpa.mse
    ORDER BY bpa.mse ASC
    LIMIT 200
"""
if selected_id is None:
    backtest_groups_df = run_query(_BT_SQL.format(where=""))
else:
    backtest_groups_df = run_query(
        _BT_SQL.format(where="WHERE b.location_id = %s"),
        (selected_id,),
    )

bt_checked_keys: list[str] = []
bt_info: dict[str, dict] = {}
if backtest_groups_df.empty:
    st.caption(
        "No backtest groups yet — they appear after the daily DAG's "
        "`walk_forward` task runs and its anchor targets get backfilled with actuals."
    )
else:
    # Always initialize on_graph to False. data_editor owns its own check
    # state via the `key` (it accumulates user edits as a diff). Pre-filling
    # from session_state confuses the diff tracking and caused clicks to need
    # a 2nd press to register. Sync is now one-way: table -> combobox.
    # Key scoped to marz so switching regions doesn't apply old row-index
    # edits to a freshly-filtered row set.
    table_view = backtest_groups_df.copy()
    table_view.insert(0, "on_graph", False)
    edited = st.data_editor(
        table_view,
        column_config={
            "on_graph": st.column_config.CheckboxColumn(
                "On graph", help="Plot this group's forecast on the chart above."
            ),
            "region": st.column_config.TextColumn("Region"),
            "prediction_made_at": st.column_config.DatetimeColumn(
                "Anchor", format="MMM DD, HH:mm"
            ),
            "n_hours": st.column_config.NumberColumn("Hours", format="%d"),
            "mse": st.column_config.NumberColumn("MSE", format="%.4f"),
            "seq_in": st.column_config.NumberColumn(
                "seq_in",
                format="%d",
                help="hours of context fed to the LSTM",
            ),
            "location_id": None,
            "model_version": None,
        },
        disabled=["region", "prediction_made_at", "n_hours", "mse", "seq_in"],
        hide_index=True,
        use_container_width=True,
        height=320,
        key=f"backtest_table_{selected_id or 'all'}",
    )
    for _, r in edited.iterrows():
        key = _encode_bt(
            int(r["location_id"]), str(r["model_version"]), r["prediction_made_at"]
        )
        bt_info[key] = {
            "location_id": int(r["location_id"]),
            "model_version": str(r["model_version"]),
            "prediction_made_at": r["prediction_made_at"],
            "region": str(r["region"]),
            "mse": float(r["mse"]),
        }
        if bool(r["on_graph"]):
            bt_checked_keys.append(key)

# 3. Sync session_state so the combobox visually reflects table changes.
COMBO_KEY = "snapshot_selector"
prev_state = list(st.session_state.get(COMBO_KEY, []))
# Keep DAG entries still valid; replace all bt entries with what's checked now.
prev_dag = [k for k in prev_state if not _is_bt(k) and k in dag_options]
# Nothing is pre-selected on initial load — user picks what to plot.
st.session_state[COMBO_KEY] = prev_dag + bt_checked_keys

# 4. Combobox — options include all DAG snapshots + any checked backtest entries.
combo_options = dag_options + bt_checked_keys


def _format_key(key: str) -> str:
    if _is_bt(key):
        info = bt_info.get(key)
        if info is None:
            return key
        anchor = pd.Timestamp(info["prediction_made_at"]).strftime("%b %d, %H:%M")
        return f"BT {info['region']} · {anchor} (MSE {info['mse']:.3f})"
    return _format_snapshot(_decode_dag_pma(key), int(dag_n_hours.get(key, 0)))


selected_keys = st.multiselect(
    "Snapshots on graph",
    options=combo_options,
    key=COMBO_KEY,
    format_func=_format_key,
    help=(
        "DAG snapshots are listed here; check rows in the table below to add "
        "backtest groups (they'll appear here too)."
    ),
)

# 5. Split selection back into the two plot paths.
#    The newest DAG snapshot is always plotted regardless of combobox state.
_combo_dag_pmas = [_decode_dag_pma(k) for k in selected_keys if not _is_bt(k)]
selected_snapshots = list(
    dict.fromkeys(
        ([newest_dag_pma] if newest_dag_pma is not None else []) + _combo_dag_pmas
    )
)
selected_bt_infos = [bt_info[k] for k in selected_keys if _is_bt(k) and k in bt_info]

# --- Predictions: every row for every selected snapshot ---
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
            FROM predictions
            WHERE prediction_made_at = ANY(%s)
            GROUP BY target_time, prediction_made_at
            ORDER BY prediction_made_at, target_time
            """,
            (selected_snapshots,),
        )
    else:
        preds = run_query(
            """
            SELECT target_time, predicted_value, actual_value, prediction_made_at
            FROM predictions
            WHERE location_id = %s AND prediction_made_at = ANY(%s)
            ORDER BY prediction_made_at, target_time
            """,
            (selected_id, selected_snapshots),
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
# One-shot: when clicked, zoom the chart to the predictions window with 1 day
# of context on the left. Any other interaction returns to the full range.
autofit_now = st.button("Autofit to predictions")

pred_x_min: pd.Timestamp | None = None
pred_x_max: pd.Timestamp | None = None
pred_y_lo: float | None = None
pred_y_hi: float | None = None


def _extend_y(values) -> None:
    global pred_y_lo, pred_y_hi
    s = pd.Series(values).dropna()
    if s.empty:
        return
    lo, hi = float(s.min()), float(s.max())
    pred_y_lo = lo if pred_y_lo is None else min(pred_y_lo, lo)
    pred_y_hi = hi if pred_y_hi is None else max(pred_y_hi, hi)


if not preds.empty:
    pred_x_min = pd.Timestamp(preds["target_time"].min())
    pred_x_max = pd.Timestamp(preds["target_time"].max())
    _extend_y(preds["predicted_value"])
    _extend_y(preds["actual_value"])

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
        snap_label = _format_snapshot(
            snap, int(dag_n_hours.get(_encode_dag(snap), len(group)))
        )
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
            int(dag_n_hours.get(_encode_dag(snaps_newest_first[0]), 0)),
        )
        st.caption(f"{len(snaps_newest_first)} snapshot(s) · latest: {latest_label}")

# --- Backtest-group overlays (one trace per checked-in-table entry) ---
if selected_bt_infos:
    bt_palette = px.colors.qualitative.Set2  # distinct from the snapshot palette
    for j, info in enumerate(selected_bt_infos):
        loc_id = info["location_id"]
        mv = info["model_version"]
        pma = info["prediction_made_at"]
        bt_pred = run_query(
            """
            SELECT target_time, predicted_value, actual_value
            FROM predictions
            WHERE location_id = %s
              AND model_version = %s
              AND prediction_made_at = %s
            ORDER BY target_time
            """,
            (loc_id, mv, pma),
        )
        if bt_pred.empty:
            continue
        bt_pred = bt_pred.copy()
        bt_pred["target_time"] = _to_naive_utc(bt_pred["target_time"])
        bt_min = pd.Timestamp(bt_pred["target_time"].min())
        bt_max = pd.Timestamp(bt_pred["target_time"].max())
        pred_x_min = bt_min if pred_x_min is None else min(pred_x_min, bt_min)
        pred_x_max = bt_max if pred_x_max is None else max(pred_x_max, bt_max)
        _extend_y(bt_pred["predicted_value"])
        _extend_y(bt_pred["actual_value"])
        bt_hover = []
        for _, r in bt_pred.iterrows():
            pred = float(r["predicted_value"])
            actual = r["actual_value"]
            if actual is not None and pd.notna(actual):
                actual_f = float(actual)
                bt_hover.append(
                    f"Predicted: {pred:.2f} °C<br>"
                    f"Actual: {actual_f:.2f} °C<br>"
                    f"Error: {pred - actual_f:+.2f} °C"
                )
            else:
                bt_hover.append(f"Predicted: {pred:.2f} °C")
        anchor_label = pd.Timestamp(pma).strftime("%b %d, %H:%M")
        trace_name = f"BT {info['region']} · {anchor_label} (MSE {info['mse']:.3f})"
        # Star anchor: actual temperature at the backtest's anchor (= pma).
        # The actuals trace above may not cover this far back, so query the
        # cell's specific observation rather than reusing `actuals`.
        x_vals = list(bt_pred["target_time"])
        y_vals = list(bt_pred["predicted_value"])
        bt_sizes = [6] * len(x_vals)
        bt_symbols = ["circle"] * len(x_vals)
        anchor_actual_df = run_query(
            """
            SELECT f.temperature_2m
            FROM fact_weather_observations f
            JOIN dim_time t ON t.time_id = f.time_id
            WHERE f.location_id = %s AND t.observed_at = %s
            """,
            (loc_id, pma),
        )
        if not anchor_actual_df.empty:
            anchor_ts_naive = pd.Timestamp(pma)
            if anchor_ts_naive.tz is not None:
                anchor_ts_naive = anchor_ts_naive.tz_convert("UTC").tz_localize(None)
            x_vals = [anchor_ts_naive] + x_vals
            y_vals = [float(anchor_actual_df["temperature_2m"].iloc[0])] + y_vals
            bt_hover = ["Train end (model input)"] + bt_hover
            bt_sizes = [15] + bt_sizes
            bt_symbols = ["star"] + bt_symbols
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                name=trace_name,
                mode="lines+markers",
                line=dict(color=bt_palette[j % len(bt_palette)], dash="dot"),
                marker=dict(size=bt_sizes, symbol=bt_symbols),
                text=bt_hover,
                hovertemplate=(
                    "<b>%{x|%b %d, %H:%M}</b><br>"
                    "%{text}"
                    f"<extra>{trace_name}</extra>"
                ),
            )
        )
train_end_ts: pd.Timestamp | None = None
if meta and meta.get("data_range_end"):
    try:
        train_end_ts = pd.Timestamp(meta["data_range_end"])
        if train_end_ts.tz is not None:
            train_end_ts = train_end_ts.tz_convert("UTC").tz_localize(None)
    except (ValueError, TypeError):
        train_end_ts = None
if train_end_ts is not None:
    # plotly's add_vline annotation does internal arithmetic on x that breaks
    # on pd.Timestamp / datetime64; pass an ISO string + add the annotation
    # separately so we don't trip its `_mean(X)` path.
    train_end_x = train_end_ts.isoformat()
    fig.add_vline(
        x=train_end_x,
        line=dict(color="#888", dash="dash", width=1),
    )
    fig.add_annotation(
        x=train_end_x,
        y=1,
        yref="paper",
        text="Train end",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=11, color="#666"),
    )

fig.update_layout(
    xaxis_title="time (UTC)",
    yaxis_title="temperature (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=640,
    dragmode="pan",
)
if autofit_now and pred_x_min is not None and pred_x_max is not None:
    x_lo = pred_x_min - pd.Timedelta(days=1)
    fig.update_xaxes(range=[x_lo, pred_x_max])
    # Include actuals within the visible window so y-bounds aren't dominated
    # by historical extremes far outside the autofit range.
    if not actuals.empty:
        in_window = actuals[
            (actuals["observed_at"] >= x_lo) & (actuals["observed_at"] <= pred_x_max)
        ]
        _extend_y(in_window["temperature_2m"])
    if pred_y_lo is not None and pred_y_hi is not None:
        pad = max(0.5, (pred_y_hi - pred_y_lo) * 0.08)
        fig.update_yaxes(range=[pred_y_lo - pad, pred_y_hi + pad])
st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True},
)

if actuals.empty and preds.empty:
    scope = "any region" if selected_id is None else selected_region
    st.info(
        f"No observations or predictions for {scope} yet. "
        "Wait for the daily Airflow DAG (or trigger it from http://localhost:8081)."
    )
