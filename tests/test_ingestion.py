"""Phase 1 unit tests — no network calls, no Docker."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


from ingestion.grid import ARMENIA, BoundingBox, make_grid
from ingestion.run_ingest import _archive_key, _year_chunks
from ingestion.schemas import HOURLY_VARIABLE_NAMES, OpenMeteoResponse
from ingestion.storage import LocalBronzeStorage


def test_grid_count_and_corners():
    grid = make_grid(ARMENIA, size=10)
    assert len(grid) == 100
    corners = {(p.lat, p.lon) for p in grid}
    assert (round(ARMENIA.lat_min, 4), round(ARMENIA.lon_min, 4)) in corners
    assert (round(ARMENIA.lat_max, 4), round(ARMENIA.lon_max, 4)) in corners


def test_grid_size_two_returns_corners_only():
    bbox = BoundingBox(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=2.0)
    grid = make_grid(bbox, size=2)
    assert len(grid) == 4
    coords = {(p.lat, p.lon) for p in grid}
    assert coords == {(0.0, 0.0), (0.0, 2.0), (1.0, 0.0), (1.0, 2.0)}


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
    from ingestion.grid import GridPoint

    key = _archive_key("armenia", 2024, GridPoint(lat=40.0700, lon=44.5000))
    assert (
        key
        == "bronze/region=armenia/dataset=archive/year=2024/lat=40.0700_lon=44.5000.json"
    )


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


def test_local_storage_writes_json(tmp_path: Path):
    storage = LocalBronzeStorage(tmp_path / "lake")
    path = storage.write_json("bronze/foo/bar.json", {"a": 1, "b": [1, 2, 3]})
    written = Path(path)
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}
