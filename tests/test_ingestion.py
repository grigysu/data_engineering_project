"""Phase 1 unit tests — no network calls, no Docker."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


from ingestion.grid import GridPoint, load_locations
from ingestion.run_ingest import (
    _archive_key,
    _chunk_has_missing,
    _forecast_key,
    _year_chunks,
)
from ingestion.schemas import HOURLY_VARIABLE_NAMES, OpenMeteoResponse


def test_load_locations_returns_one_grid_point_per_entry(tmp_path: Path):
    payload = [
        {"region": "Yerevan", "lat": 40.1776, "lon": 44.5126, "capital": "Yerevan"},
        {"region": "Shirak", "lat": 40.7931, "lon": 43.8464, "capital": "Gyumri"},
    ]
    f = tmp_path / "locations.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    pts = load_locations(f)
    assert len(pts) == 2
    assert pts[0].name == "Yerevan" and pts[0].lat == 40.1776
    assert pts[1].name == "Shirak" and pts[1].lon == 43.8464


def test_grid_point_cell_id_format():
    p = GridPoint(lat=40.07, lon=44.5, name="Yerevan")
    assert p.cell_id == "lat=40.0700/lon=44.5000"


def test_year_chunks_splits_on_calendar_boundary():
    chunks = list(_year_chunks(date(2023, 6, 1), date(2025, 3, 15)))
    assert chunks == [
        (date(2023, 6, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 3, 15)),
    ]


def test_year_chunks_single_year():
    chunks = list(_year_chunks(date(2024, 3, 1), date(2024, 9, 30)))
    assert chunks == [(date(2024, 3, 1), date(2024, 9, 30))]


def test_archive_key_layout():
    key = _archive_key(
        "armenia", 2024, GridPoint(lat=40.0700, lon=44.5000, name="Yerevan")
    )
    assert key == (
        "bronze/region=armenia/dataset=archive/marz=Yerevan"
        "/year=2024/lat=40.0700_lon=44.5000.json"
    )


def test_archive_key_handles_unnamed_point():
    key = _archive_key("armenia", 2024, GridPoint(lat=40.0700, lon=44.5000))
    assert "/marz=unknown/" in key


def test_forecast_key_includes_marz_partition():
    from datetime import datetime, timezone

    ts = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    key = _forecast_key("armenia", ts, GridPoint(lat=40.79, lon=43.85, name="Shirak"))
    assert "/marz=Shirak/" in key
    assert "run_ts=20260525T120000Z" in key


def _fake_response(n_hours: int = 3) -> dict:
    return {
        "latitude": 40.0,
        "longitude": 44.5,
        "timezone": "UTC",
        "elevation": 989.0,
        "utc_offset_seconds": 0,
        "hourly_units": {v: "" for v in HOURLY_VARIABLE_NAMES},
        "hourly": {
            "time": [f"2024-01-01T{h:02d}:00" for h in range(n_hours)],
            **{v: [0.0] * n_hours for v in HOURLY_VARIABLE_NAMES},
        },
    }


def test_schema_accepts_well_formed_response():
    parsed = OpenMeteoResponse.model_validate(_fake_response(n_hours=5))
    assert parsed.row_count() == 5
    assert len(parsed.hourly.temperature_2m) == 5


def test_schema_tolerates_nulls_in_hourly_arrays():
    payload = _fake_response(n_hours=2)
    payload["hourly"]["temperature_2m"] = [None, 1.5]
    parsed = OpenMeteoResponse.model_validate(payload)
    assert parsed.hourly.temperature_2m == [None, 1.5]


def test_chunk_has_missing_full_coverage():
    present = {date(2026, 5, d) for d in range(1, 11)}
    assert not _chunk_has_missing(date(2026, 5, 3), date(2026, 5, 7), present)


def test_chunk_has_missing_partial_coverage():
    present = {date(2026, 5, d) for d in (1, 2, 3, 6, 7)}
    assert _chunk_has_missing(date(2026, 5, 1), date(2026, 5, 7), present)


def test_chunk_has_missing_empty_coverage():
    assert _chunk_has_missing(date(2026, 5, 1), date(2026, 5, 1), set())
