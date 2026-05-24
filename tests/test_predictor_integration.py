"""End-to-end tests for the Predictor against a real checkpoint + gold parquet.

Opt-in: set RUN_INTEGRATION_TESTS=1 to enable. Requires:
  - docker-compose up (MinIO + Postgres reachable)
  - At least one training run (so checkpoints/best.pt exists)
  - Gold parquet at s3://weather-lake/gold/weather_features

Streamlit imports Predictor directly (no HTTP), so this test exercises exactly
the path the dashboard uses.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CHECKPOINT = Path("checkpoints/best.pt")
GOLD = "s3://weather-lake/gold/weather_features"


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1" or not CHECKPOINT.exists(),
    reason=(
        "predictor integration tests: set RUN_INTEGRATION_TESTS=1, run a training "
        "job, and ensure docker-compose is up."
    ),
)


def test_predictor_snaps_to_nearest_grid_point():
    from serving.predictor import Predictor

    p = Predictor(CHECKPOINT, GOLD, persist=False)
    points = p.grid_points()
    assert len(points) > 0

    target_lat, target_lon = points[0]
    snapped = p._nearest_grid_point(target_lat + 0.001, target_lon - 0.001)
    assert snapped == (target_lat, target_lon)


def test_predictor_returns_well_formed_forecast():
    from serving.predictor import Predictor

    p = Predictor(CHECKPOINT, GOLD, persist=False)
    lat, lon = p.grid_points()[0]
    resp = p.predict(lat, lon)

    assert resp.snapped_lat == lat
    assert resp.snapped_lon == lon
    assert resp.horizon_hours == p.spec.seq_out
    assert len(resp.predictions) == p.spec.seq_out
    assert resp.predictions[0].hours_ahead == 1
    assert resp.predictions[-1].hours_ahead == p.spec.seq_out

    # Plausible Celsius range for Armenia.
    for p_i in resp.predictions:
        assert -50.0 < p_i.temperature_2m_c < 60.0
