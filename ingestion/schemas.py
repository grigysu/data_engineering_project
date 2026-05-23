"""Pydantic schemas for Open-Meteo responses.

We validate the *shape* at ingestion time so schema drift surfaces immediately
rather than blowing up downstream Spark jobs days later. Bronze is still
stored as raw JSON — these models are for in-process validation only.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# Hourly variables we request from Open-Meteo. Keep this list in sync with
# `HOURLY_VARIABLES` in openmeteo_client.py — the schema below mirrors it.
HOURLY_VARIABLE_NAMES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
)


class HourlyBlock(BaseModel):
    """Time-aligned arrays — `time[i]` corresponds to `temperature_2m[i]`, etc."""

    model_config = ConfigDict(extra="allow")

    time: list[str]
    temperature_2m: list[Optional[float]]
    relative_humidity_2m: list[Optional[float]]
    dew_point_2m: list[Optional[float]]
    precipitation: list[Optional[float]]
    pressure_msl: list[Optional[float]]
    cloud_cover: list[Optional[float]]
    wind_speed_10m: list[Optional[float]]
    wind_direction_10m: list[Optional[float]]
    wind_gusts_10m: list[Optional[float]]
    shortwave_radiation: list[Optional[float]]


class OpenMeteoResponse(BaseModel):
    """Top-level Open-Meteo response (archive + forecast share this shape)."""

    model_config = ConfigDict(extra="allow")

    latitude: float
    longitude: float
    timezone: str
    elevation: Optional[float] = None
    utc_offset_seconds: Optional[int] = None
    hourly_units: dict[str, str] = Field(default_factory=dict)
    hourly: HourlyBlock

    def row_count(self) -> int:
        return len(self.hourly.time)
