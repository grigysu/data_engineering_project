"""End-to-end tests for the predictor + FastAPI serving layer.

Builds a real Predictor against the actual on-disk checkpoint + MinIO-hosted
gold parquet. Skipped by default; opt in with `RUN_SERVING_TESTS=1` once
docker-compose is up and the LSTM has been trained at least once.

(FastAPI is scheduled to be replaced by Streamlit in Phase 2c; this file
will be retired then.)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CHECKPOINT = Path("checkpoints/best.pt")
GOLD = "s3://weather-lake/gold/weather_features"


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SERVING_TESTS") != "1" or not CHECKPOINT.exists(),
    reason=(
        "serving integration tests: set RUN_SERVING_TESTS=1 and ensure "
        "checkpoints/best.pt exists + MinIO is reachable."
    ),
)


def test_predictor_snaps_to_nearest_grid_point():
    from serving.predictor import Predictor

    p = Predictor(CHECKPOINT, GOLD)
    points = p.grid_points()
    assert len(points) > 0

    # Pick a real grid point and offset it by a small (lat, lon) jitter;
    # snap must return the exact original.
    target_lat, target_lon = points[0]
    snapped = p._nearest_grid_point(target_lat + 0.001, target_lon - 0.001)
    assert snapped == (target_lat, target_lon)


def test_predictor_returns_well_formed_forecast():
    from serving.predictor import Predictor

    p = Predictor(CHECKPOINT, GOLD)
    lat, lon = p.grid_points()[0]
    resp = p.predict(lat, lon)

    assert resp.snapped_lat == lat
    assert resp.snapped_lon == lon
    assert resp.horizon_hours == p.spec.seq_out
    assert len(resp.predictions) == p.spec.seq_out
    assert resp.predictions[0].hours_ahead == 1
    assert resp.predictions[-1].hours_ahead == p.spec.seq_out

    # Temperatures should be in a physically plausible Celsius range for
    # Armenia — wide bounds because the model is undertrained and may
    # output anything sane-ish.
    for p_i in resp.predictions:
        assert -50.0 < p_i.temperature_2m_c < 60.0


def test_api_health_returns_ok():
    from fastapi.testclient import TestClient

    from serving.api import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["seq_in_hours"] > 0
        assert body["horizon_hours"] > 0
        assert body["grid_size"] > 0


def test_api_forecast_returns_predictions():
    from fastapi.testclient import TestClient

    from serving.api import app

    with TestClient(app) as client:
        grid = client.get("/grid_points").json()
        first = grid["points"][0]
        r = client.post("/forecast", json={"lat": first["lat"], "lon": first["lon"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["snapped_lat"] == first["lat"]
        assert body["snapped_lon"] == first["lon"]
        assert len(body["predictions"]) == body["horizon_hours"]
        # Each prediction is a dict {hours_ahead, temperature_2m_c}
        assert {"hours_ahead", "temperature_2m_c"} <= set(body["predictions"][0].keys())


def test_api_forecast_rejects_out_of_range_lat():
    from fastapi.testclient import TestClient

    from serving.api import app

    with TestClient(app) as client:
        r = client.post("/forecast", json={"lat": 999.0, "lon": 0.0})
        assert r.status_code == 422  # pydantic validation
