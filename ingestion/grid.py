"""Grid of points to ingest, loaded from a curated locations file.

Previously a uniform 10×10 bbox sweep; now one point per Armenian admin-1
unit (10 marzes + Yerevan city), each anchored at the marz capital city.
Coordinates come from Open-Meteo's geocoding API and are materialized to
`config/locations.json` by `ingestion/seed_locations.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class GridPoint:
    lat: float
    lon: float
    name: str | None = None  # marz name, e.g. "Yerevan", "Shirak". Optional.

    @property
    def cell_id(self) -> str:
        return f"lat={self.lat:.4f}/lon={self.lon:.4f}"


LOCATIONS_FILE = Path(__file__).resolve().parents[1] / "config" / "locations.json"


def load_locations(path: Path = LOCATIONS_FILE) -> list[GridPoint]:
    """Read `locations.json` and return one GridPoint per entry.

    The file is produced by `python -m ingestion.seed_locations`. Each entry
    has at least `region` (marz name), `lat`, `lon`.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        GridPoint(
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            name=entry["region"],
        )
        for entry in payload
    ]


def default_grid() -> list[GridPoint]:
    return load_locations()


def iter_chunks(points: list[GridPoint], chunk_size: int) -> Iterator[list[GridPoint]]:
    """Yield successive `chunk_size` slices of `points`."""
    for i in range(0, len(points), chunk_size):
        yield points[i : i + chunk_size]


if __name__ == "__main__":
    grid = default_grid()
    print(f"{len(grid)} points:")
    for p in grid:
        print(f"  {p.name:>14s}  {p.cell_id}")
