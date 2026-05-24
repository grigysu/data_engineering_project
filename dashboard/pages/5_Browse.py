"""Browse — pre-canned warehouse summaries + a link to Adminer."""

from __future__ import annotations

import os

import streamlit as st

from dashboard.state import run_query, sidebar_config

sidebar_config()
st.title("Browse")
st.caption(
    "Quick read-only summaries from the warehouse. For ad-hoc SQL, jump to Adminer."
)

ADMINER_URL = os.getenv("ADMINER_URL", "http://localhost:8082")
st.link_button("Open Adminer", ADMINER_URL, type="primary")
st.caption(
    "Adminer connection: server `postgres`, user `weather`, db `weather_dw`. "
    "Password is in `.env`."
)

st.divider()

st.subheader("Rows per day")
df = run_query(
    """
    SELECT t.date, COUNT(*) AS rows
    FROM fact_weather_observations f
    JOIN dim_time t ON t.time_id = f.time_id
    GROUP BY t.date
    ORDER BY t.date DESC
    LIMIT 60
    """
)
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Predictions per model")
df = run_query(
    """
    SELECT model_version,
           COUNT(*) AS total,
           COUNT(actual_value) AS with_actual,
           AVG(ABS(predicted_value - actual_value))
               FILTER (WHERE actual_value IS NOT NULL) AS mae_c
    FROM predictions
    GROUP BY model_version
    ORDER BY MAX(prediction_made_at) DESC
    """
)
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Grid cells with the most data")
df = run_query(
    """
    SELECT l.lat, l.lon, l.elevation, COUNT(*) AS observations
    FROM fact_weather_observations f
    JOIN dim_location l ON l.location_id = f.location_id
    GROUP BY l.lat, l.lon, l.elevation
    ORDER BY observations DESC
    LIMIT 20
    """
)
st.dataframe(df, use_container_width=True, hide_index=True)
