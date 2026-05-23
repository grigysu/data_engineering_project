"""Lat/lon grid generation for the configured region."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GridPoint:
    lat: float
    lon: float

    @property
    def cell_id(self) -> str:
        return f"lat={self.lat:.4f}/lon={self.lon:.4f}"


@dataclass(frozen=True)
class BoundingBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        assert self.lat_min < self.lat_max, "lat_min must be < lat_max"
        assert self.lon_min < self.lon_max, "lon_min must be < lon_max"


# Default: Armenia. Override via env vars (see .env.example).
ARMENIA = BoundingBox(lat_min=38.84, lat_max=41.30, lon_min=43.45, lon_max=46.63)


def load_bbox_from_env() -> BoundingBox:
    return BoundingBox(
        lat_min=float(os.getenv("GRID_LAT_MIN", ARMENIA.lat_min)),
        lat_max=float(os.getenv("GRID_LAT_MAX", ARMENIA.lat_max)),
        lon_min=float(os.getenv("GRID_LON_MIN", ARMENIA.lon_min)),
        lon_max=float(os.getenv("GRID_LON_MAX", ARMENIA.lon_max)),
    )


def make_grid(bbox: BoundingBox, size: int) -> list[GridPoint]:
    """Return a size x size evenly-spaced grid covering bbox (inclusive of corners)."""
    assert size >= 2, "grid size must be >= 2 (need at least the corner points)"
    lat_step = (bbox.lat_max - bbox.lat_min) / (size - 1)
    lon_step = (bbox.lon_max - bbox.lon_min) / (size - 1)
    points: list[GridPoint] = []
    for i in range(size):
        for j in range(size):
            points.append(
                GridPoint(
                    lat=round(bbox.lat_min + i * lat_step, 4),
                    lon=round(bbox.lon_min + j * lon_step, 4),
                )
            )
    return points


def default_grid() -> list[GridPoint]:
    bbox = load_bbox_from_env()
    size = int(os.getenv("GRID_SIZE", "10"))
    return make_grid(bbox, size)


def iter_chunks(points: list[GridPoint], chunk_size: int) -> Iterator[list[GridPoint]]:
    """Yield successive `chunk_size` slices of `points`."""
    for i in range(0, len(points), chunk_size):
        yield points[i : i + chunk_size]


if __name__ == "__main__":
    grid = default_grid()
    print(f"{len(grid)} points covering {os.getenv('REGION_NAME', 'armenia')}:")
    for p in grid[:5]:
        print(f"  {p.cell_id}")
    print("  ...")
