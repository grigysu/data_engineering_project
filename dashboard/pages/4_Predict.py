"""Predict — pick (lat, lon), see the forecast, compare with later actuals."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.state import (
    fetch_best_model,
    get_predictor,
    run_query,
    sidebar_config,
)

cfg = sidebar_config()
st.title("Predict")
st.caption(
    "Forecast a `seq_out`-hour temperature trajectory for any grid cell. Each "
    "forecast is persisted to Postgres so the comparison tab can show how the "
    "model has done historically."
)

try:
    predictor = get_predictor(cfg["checkpoint"], cfg["gold"])
except Exception as exc:
    st.error(
        f"Predictor failed to load (checkpoint=`{cfg['checkpoint']}` "
        f"gold=`{cfg['gold']}`): {exc}"
    )
    st.stop()

points = predictor.grid_points()
if not points:
    st.warning("Gold parquet has no points yet — backfill on the **Data** page.")
    st.stop()

best = fetch_best_model()
if best:
    st.caption(
        f"Model `{best.model_version}` · trained on "
        f"`{best.data_range_start} → {best.data_range_end}` · "
        f"val MSE {best.best_val_mse:.4f}"
    )

tab1, tab2 = st.tabs(["Forecast now", "Predicted vs. actual (history)"])


# --- Forecast tab ---
with tab1:
    grid_df = pd.DataFrame(points, columns=["lat", "lon"])
    selected = st.selectbox(
        "Grid cell",
        options=range(len(grid_df)),
        format_func=lambda i: f"({grid_df.iloc[i, 0]:.4f}, {grid_df.iloc[i, 1]:.4f})",
    )
    lat = float(grid_df.iloc[selected, 0])
    lon = float(grid_df.iloc[selected, 1])

    if st.button("Forecast", type="primary"):
        try:
            resp = predictor.predict(lat, lon)
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            st.stop()
        st.success(
            f"Anchor `{resp.forecast_anchor}` · horizon {resp.horizon_hours}h · "
            f"persisted: **{resp.persisted}**"
        )
        forecast_df = pd.DataFrame(
            [
                {"hours_ahead": p.hours_ahead, "temperature_C": p.temperature_2m_c}
                for p in resp.predictions
            ]
        )
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        fig = px.line(
            forecast_df,
            x="hours_ahead",
            y="temperature_C",
            markers=True,
            title=f"{resp.horizon_hours}-hour forecast — ({lat:.4f}, {lon:.4f})",
        )
        st.plotly_chart(fig, use_container_width=True)


# --- History tab ---
with tab2:
    st.caption(
        "Joins `predictions` to `dim_location`. Rows where `actual_value IS NOT "
        "NULL` have been backfilled by the nightly Airflow task; the rest are "
        "still pending observations."
    )
    cell_pick = st.selectbox(
        "Grid cell",
        options=range(len(grid_df)),
        key="history_cell",
        format_func=lambda i: f"({grid_df.iloc[i, 0]:.4f}, {grid_df.iloc[i, 1]:.4f})",
    )
    history_lat = float(grid_df.iloc[cell_pick, 0])
    history_lon = float(grid_df.iloc[cell_pick, 1])

    history = run_query(
        """
        SELECT p.target_time, p.predicted_value, p.actual_value, p.model_version
        FROM predictions p
        JOIN dim_location l ON l.location_id = p.location_id
        WHERE l.lat = %s AND l.lon = %s
        ORDER BY p.target_time DESC
        LIMIT 500
        """,
        (history_lat, history_lon),
    )
    if history.empty:
        st.info("No predictions on file for this cell yet. Run a forecast above first.")
    else:
        with_actual = history.dropna(subset=["actual_value"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Predictions", len(history))
        col2.metric("Backfilled actuals", len(with_actual))
        if not with_actual.empty:
            err = (
                (with_actual["predicted_value"] - with_actual["actual_value"])
                .abs()
                .mean()
            )
            col3.metric("Mean abs error (°C)", f"{err:.2f}")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["target_time"],
                y=history["predicted_value"],
                name="Predicted",
                mode="markers",
            )
        )
        if not with_actual.empty:
            fig.add_trace(
                go.Scatter(
                    x=with_actual["target_time"],
                    y=with_actual["actual_value"],
                    name="Actual",
                    mode="markers",
                )
            )
        fig.update_layout(
            title=f"Predicted vs. actual — ({history_lat:.4f}, {history_lon:.4f})",
            xaxis_title="target time",
            yaxis_title="temperature (°C)",
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Raw rows"):
            st.dataframe(history, use_container_width=True, hide_index=True)
