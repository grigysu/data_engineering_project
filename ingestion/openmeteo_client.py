"""Async client for the Open-Meteo Historical Archive and Forecast APIs.

Docs:
  - Archive: https://open-meteo.com/en/docs/historical-weather-api
  - Forecast: https://open-meteo.com/en/docs

Free tier is generous for our 100-point grid:
  - Archive: long date ranges allowed in a single call → ~1 call per point per year.
  - Forecast: up to 16 days ahead, hourly.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import httpx

from ingestion.schemas import HOURLY_VARIABLE_NAMES, OpenMeteoResponse


HOURLY_PARAM = ",".join(HOURLY_VARIABLE_NAMES)


class OpenMeteoClient:
    """Thin async wrapper around Open-Meteo. Bounded retries on transient errors."""

    def __init__(
        self,
        archive_url: str,
        forecast_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.archive_url = archive_url
        self.forecast_url = forecast_url
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "weather-data-engine/0.1"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenMeteoClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def fetch_archive(
        self, lat: float, lon: float, start: date, end: date
    ) -> OpenMeteoResponse:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": HOURLY_PARAM,
            "timezone": "UTC",
        }
        raw = await self._get_with_retry(self.archive_url, params)
        return OpenMeteoResponse.model_validate(raw)

    async def fetch_forecast(
        self, lat: float, lon: float, forecast_days: int = 14
    ) -> OpenMeteoResponse:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": HOURLY_PARAM,
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        raw = await self._get_with_retry(self.forecast_url, params)
        return OpenMeteoResponse.model_validate(raw)

    async def _get_with_retry(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                # Retry only on 429 (rate limit) and 5xx; bail on other 4xx.
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    if status != 429 and status < 500:
                        raise
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
        assert last_exc is not None
        raise last_exc
