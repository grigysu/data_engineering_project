"""CLI: ingest Open-Meteo data into the bronze layer.

Examples:
    # Backfill historical hourly data for the whole grid:
    python -m ingestion.run_ingest --backfill 2023-01-01:2024-12-31

    # Pull the latest 14-day forecast for every grid point:
    python -m ingestion.run_ingest --forecast --forecast-days 14

    # Smoke test: just one year, lower concurrency:
    python -m ingestion.run_ingest --backfill 2024-01-01:2024-12-31 --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

from dotenv import load_dotenv

from ingestion.grid import GridPoint, default_grid
from ingestion.healthcheck import OK, check_openmeteo, render_text
from ingestion.openmeteo_client import OpenMeteoClient
from ingestion.schemas import OpenMeteoResponse
from ingestion.storage import BronzeStorage, storage_from_env

load_dotenv()


@dataclass(frozen=True)
class IngestStats:
    requested: int = 0
    succeeded: int = 0
    failed: int = 0
    rows: int = 0


def _archive_key(region: str, year: int, point: GridPoint) -> str:
    return (
        f"bronze/region={region}/dataset=archive/year={year}"
        f"/lat={point.lat:.4f}_lon={point.lon:.4f}.json"
    )


def _forecast_key(region: str, ts: datetime, point: GridPoint) -> str:
    return (
        f"bronze/region={region}/dataset=forecast"
        f"/run_ts={ts.strftime('%Y%m%dT%H%M%SZ')}"
        f"/lat={point.lat:.4f}_lon={point.lon:.4f}.json"
    )


def _year_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Split [start, end] into per-calendar-year sub-ranges (inclusive)."""
    cursor = start
    while cursor <= end:
        year_end = date(cursor.year, 12, 31)
        chunk_end = min(year_end, end)
        yield cursor, chunk_end
        cursor = date(cursor.year + 1, 1, 1)


def _parse_window(spec: str) -> tuple[date, date]:
    try:
        s, e = spec.split(":")
        return date.fromisoformat(s), date.fromisoformat(e)
    except ValueError as exc:
        raise SystemExit(
            f"--backfill must be YYYY-MM-DD:YYYY-MM-DD (got {spec!r})"
        ) from exc


async def _ingest_one_archive(
    client: OpenMeteoClient,
    storage: BronzeStorage,
    sem: asyncio.Semaphore,
    region: str,
    point: GridPoint,
    start: date,
    end: date,
) -> tuple[bool, int]:
    async with sem:
        try:
            response: OpenMeteoResponse = await client.fetch_archive(
                point.lat, point.lon, start, end
            )
        except Exception as exc:
            print(
                f"  !! archive {point.cell_id} {start}..{end}: {exc}", file=sys.stderr
            )
            return False, 0
        key = _archive_key(region, start.year, point)
        storage.write_json(key, response.model_dump(mode="json"))
        return True, response.row_count()


async def _ingest_one_forecast(
    client: OpenMeteoClient,
    storage: BronzeStorage,
    sem: asyncio.Semaphore,
    region: str,
    run_ts: datetime,
    point: GridPoint,
    forecast_days: int,
) -> tuple[bool, int]:
    async with sem:
        try:
            response = await client.fetch_forecast(point.lat, point.lon, forecast_days)
        except Exception as exc:
            print(f"  !! forecast {point.cell_id}: {exc}", file=sys.stderr)
            return False, 0
        key = _forecast_key(region, run_ts, point)
        storage.write_json(key, response.model_dump(mode="json"))
        return True, response.row_count()


async def run_backfill(
    client: OpenMeteoClient,
    storage: BronzeStorage,
    grid: list[GridPoint],
    region: str,
    start: date,
    end: date,
    concurrency: int,
) -> IngestStats:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _ingest_one_archive(client, storage, sem, region, p, cs, ce)
        for p in grid
        for cs, ce in _year_chunks(start, end)
    ]
    print(
        f"Backfilling {len(grid)} points × {sum(1 for _ in _year_chunks(start, end))} years "
        f"= {len(tasks)} requests (concurrency={concurrency})"
    )
    results = await asyncio.gather(*tasks)
    succeeded = sum(1 for ok, _ in results if ok)
    rows = sum(n for ok, n in results if ok)
    return IngestStats(
        requested=len(tasks),
        succeeded=succeeded,
        failed=len(tasks) - succeeded,
        rows=rows,
    )


async def run_forecast(
    client: OpenMeteoClient,
    storage: BronzeStorage,
    grid: list[GridPoint],
    region: str,
    forecast_days: int,
    concurrency: int,
) -> IngestStats:
    sem = asyncio.Semaphore(concurrency)
    run_ts = datetime.now(timezone.utc).replace(microsecond=0)
    tasks = [
        _ingest_one_forecast(client, storage, sem, region, run_ts, p, forecast_days)
        for p in grid
    ]
    print(
        f"Forecasting {len(grid)} points × {forecast_days}d (concurrency={concurrency})"
    )
    results = await asyncio.gather(*tasks)
    succeeded = sum(1 for ok, _ in results if ok)
    rows = sum(n for ok, n in results if ok)
    return IngestStats(
        requested=len(tasks),
        succeeded=succeeded,
        failed=len(tasks) - succeeded,
        rows=rows,
    )


async def _preflight(required: list[str], skip: bool) -> bool:
    """Probe Open-Meteo before doing real work. Returns True if safe to proceed."""
    if skip:
        print("(skipping preflight healthcheck, --skip-check set)")
        return True
    print("Preflight: probing Open-Meteo endpoints...")
    results = await check_openmeteo()
    print(render_text(results))
    by_name = {h.name: h for h in results}
    bad = [n for n in required if by_name[n].verdict != OK]
    if bad:
        print(
            f"\nAborting: required endpoint(s) not OK: {', '.join(bad)}. "
            f"Pass --skip-check to override.",
            file=sys.stderr,
        )
        return False
    print()
    return True


async def main_async(args: argparse.Namespace) -> None:
    region = os.getenv("REGION_NAME", "armenia")
    grid = default_grid()
    storage = storage_from_env()
    client = OpenMeteoClient(
        archive_url=os.getenv(
            "OPENMETEO_BASE_URL", "https://archive-api.open-meteo.com/v1/archive"
        ),
        forecast_url=os.getenv(
            "OPENMETEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
        ),
    )

    required: list[str] = []
    if args.backfill:
        required.append("archive")
    if args.forecast:
        required.append("forecast")
    if required and not await _preflight(required, args.skip_check):
        sys.exit(2)

    print(
        f"Region: {region}  |  Grid: {len(grid)} points  |  Storage: {storage.describe()}"
    )

    async with client:
        if args.backfill:
            start, end = _parse_window(args.backfill)
            stats = await run_backfill(
                client, storage, grid, region, start, end, args.concurrency
            )
            print(
                f"\nBackfill done: {stats.succeeded}/{stats.requested} files "
                f"({stats.failed} failed), {stats.rows:,} hourly rows."
            )
        if args.forecast:
            stats = await run_forecast(
                client, storage, grid, region, args.forecast_days, args.concurrency
            )
            print(
                f"\nForecast done: {stats.succeeded}/{stats.requested} files "
                f"({stats.failed} failed), {stats.rows:,} hourly rows."
            )
        if not args.backfill and not args.forecast:
            print(
                "Nothing to do. Pass --backfill YYYY-MM-DD:YYYY-MM-DD and/or --forecast."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Open-Meteo data into bronze.")
    parser.add_argument(
        "--backfill",
        metavar="START:END",
        help="Backfill historical hourly archive for this date range.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Pull the latest forecast for every grid point.",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=14,
        help="Forecast horizon in days (Open-Meteo max ~16). Default: 14.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("INGEST_CONCURRENCY", "5")),
        help="Max in-flight HTTP requests. Default: 5.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the preflight Open-Meteo healthcheck.",
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
