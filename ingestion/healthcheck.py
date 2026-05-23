"""Reachability + health probes for the Open-Meteo endpoints.

Always run this before launching a long backfill: if the archive endpoint is
returning 5xx, we want to fail fast with a clear diagnosis instead of burning
100+ retries.

Probes are layered so failures localize the problem:
  1. DNS resolution
  2. TCP connect on :443
  3. HTTP root request (catches CDN / origin-proxy failures)
  4. HTTP minimal valid query (catches application-layer issues)

The probe results are returned as structured objects so the CLI and the
ingestion entrypoint can both consume them.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()


# Verdicts roll up the layered probes.
OK = "OK"  # Endpoint is fully usable for ingestion.
DEGRADED = "DEGRADED"  # Reachable but returning errors (5xx, 429, etc).
DOWN = "DOWN"  # DNS / TCP failure — endpoint cannot be reached.


@dataclass
class ProbeResult:
    name: str
    success: bool
    detail: str
    elapsed_ms: float


@dataclass
class EndpointHealth:
    name: str
    url: str
    verdict: str
    probes: list[ProbeResult] = field(default_factory=list)

    def add(self, p: ProbeResult) -> None:
        self.probes.append(p)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _dns_probe(host: str) -> ProbeResult:
    t0 = _now_ms()
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
        ips = sorted({i[4][0] for i in infos})
        return ProbeResult("dns", True, ",".join(ips), _now_ms() - t0)
    except OSError as exc:
        return ProbeResult("dns", False, str(exc), _now_ms() - t0)


def _tcp_probe(host: str, port: int, timeout: float = 5.0) -> ProbeResult:
    t0 = _now_ms()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return ProbeResult("tcp", True, f"{host}:{port} reachable", _now_ms() - t0)
    except OSError as exc:
        return ProbeResult("tcp", False, str(exc), _now_ms() - t0)


async def _http_probe(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    params: Optional[dict] = None,
    accept_4xx: bool = False,
) -> ProbeResult:
    t0 = _now_ms()
    try:
        resp = await client.get(url, params=params)
        elapsed = _now_ms() - t0
        if resp.status_code == 200:
            return ProbeResult(
                name, True, f"HTTP 200 ({resp.headers.get('server', '?')})", elapsed
            )
        if accept_4xx and 400 <= resp.status_code < 500:
            return ProbeResult(
                name, True, f"HTTP {resp.status_code} (expected — endpoint up)", elapsed
            )
        return ProbeResult(
            name, False, f"HTTP {resp.status_code} {resp.reason_phrase}", elapsed
        )
    except (httpx.TransportError, httpx.HTTPError) as exc:
        return ProbeResult(name, False, f"{type(exc).__name__}: {exc}", _now_ms() - t0)


async def check_endpoint(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    minimal_params: dict,
) -> EndpointHealth:
    """Layered probe for one Open-Meteo endpoint."""
    host = urlparse(url).hostname or ""
    h = EndpointHealth(name=name, url=url, verdict=DOWN)

    dns = _dns_probe(host)
    h.add(dns)
    if not dns.success:
        return h

    tcp = _tcp_probe(host, 443)
    h.add(tcp)
    if not tcp.success:
        return h

    # Endpoint root: a 4xx here is fine (means the service is up, we just sent no params).
    root = await _http_probe(client, "http_root", url, accept_4xx=True)
    h.add(root)

    # Minimal valid query: this is the real test for application-layer health.
    query = await _http_probe(client, "http_query", url, params=minimal_params)
    h.add(query)

    if query.success:
        h.verdict = OK
    elif root.success:
        h.verdict = DEGRADED  # service up but the data path is broken
    else:
        h.verdict = DOWN
    return h


async def check_openmeteo(
    archive_url: Optional[str] = None,
    forecast_url: Optional[str] = None,
    timeout_seconds: float = 15.0,
) -> list[EndpointHealth]:
    archive_url = archive_url or os.getenv(
        "OPENMETEO_BASE_URL", "https://archive-api.open-meteo.com/v1/archive"
    )
    forecast_url = forecast_url or os.getenv(
        "OPENMETEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        headers={"User-Agent": "weather-data-engine/0.1 healthcheck"},
    ) as client:
        return await asyncio.gather(
            check_endpoint(
                client,
                "forecast",
                forecast_url,
                minimal_params={
                    "latitude": 40,
                    "longitude": 44,
                    "hourly": "temperature_2m",
                    "forecast_days": 1,
                },
            ),
            check_endpoint(
                client,
                "archive",
                archive_url,
                minimal_params={
                    "latitude": 40,
                    "longitude": 44,
                    "start_date": "2024-06-01",
                    "end_date": "2024-06-01",
                    "hourly": "temperature_2m",
                },
            ),
        )


def render_text(results: list[EndpointHealth]) -> str:
    lines: list[str] = []
    for h in results:
        lines.append(f"[{h.verdict:<8}] {h.name}  ({h.url})")
        for p in h.probes:
            mark = "OK" if p.success else "!!"
            lines.append(f"   {mark} {p.name:<10} {p.elapsed_ms:>7.0f} ms  {p.detail}")
    return "\n".join(lines)


def render_json(results: list[EndpointHealth]) -> str:
    return json.dumps([asdict(h) for h in results], indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Health-check Open-Meteo endpoints.")
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of human text."
    )
    parser.add_argument(
        "--require",
        choices=["forecast", "archive", "all"],
        default="all",
        help="Which endpoints must be OK for exit code 0. Default: all.",
    )
    args = parser.parse_args()

    results = asyncio.run(check_openmeteo())
    print(render_json(results) if args.json else render_text(results))

    by_name = {h.name: h for h in results}
    if args.require == "all":
        required = list(by_name.values())
    else:
        required = [by_name[args.require]]
    sys.exit(0 if all(h.verdict == OK for h in required) else 1)


if __name__ == "__main__":
    main()
