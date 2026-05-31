"""Seed `config/locations.json` from Open-Meteo's geocoding API.

For each Armenian admin-1 unit (10 marzes + Yerevan city), fetches the
authoritative coordinates of its capital city. Output is a JSON list keyed
by marz name — same string used in `dashboard/data/armenia_marzes.geojson`
properties.name, so the dashboard's choropleth join works.

Run once (or whenever the city list changes):
    python -m ingestion.seed_locations

Idempotent: re-running overwrites locations.json with the same payload as
long as Open-Meteo's geocoding answers stay stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# (marz_canonical_name, capital_city_name_to_geocode).
# Marz name MUST match the canonical names in dashboard/data/armenia_marzes.geojson.
MARZ_CAPITALS: list[tuple[str, str]] = [
    ("Yerevan", "Yerevan"),
    ("Aragatsotn", "Ashtarak"),
    ("Ararat", "Artashat"),
    ("Armavir", "Armavir"),
    ("Gegharkunik", "Gavar"),
    ("Kotayk", "Hrazdan"),
    ("Lori", "Vanadzor"),
    ("Shirak", "Gyumri"),
    ("Syunik", "Kapan"),
    ("Tavush", "Ijevan"),
    ("Vayots Dzor", "Yeghegnadzor"),
]

OUTPUT = Path(__file__).resolve().parents[1] / "config" / "locations.json"


def geocode(client: httpx.Client, name: str) -> dict:
    """Return the highest-population Armenian hit for `name`."""
    r = client.get(
        GEOCODE_URL,
        params={"name": name, "count": 10, "language": "en", "format": "json"},
    )
    r.raise_for_status()
    results = r.json().get("results") or []
    armenian = [
        h
        for h in results
        if h.get("country_code") == "AM" or h.get("country") == "Armenia"
    ]
    if not armenian:
        raise RuntimeError(f"geocoding: no Armenian hit for {name!r}")
    # Highest population wins (handles e.g. multiple "Armavir" cities in the world).
    armenian.sort(key=lambda h: h.get("population") or 0, reverse=True)
    return armenian[0]


def main() -> None:
    out: list[dict] = []
    with httpx.Client(timeout=15.0) as client:
        for marz, capital in MARZ_CAPITALS:
            print(f"  [{marz}] geocoding {capital!r} ...", file=sys.stderr)
            hit = geocode(client, capital)
            out.append(
                {
                    "region": marz,
                    "capital": hit["name"],
                    "lat": round(float(hit["latitude"]), 4),
                    "lon": round(float(hit["longitude"]), 4),
                    "elevation": float(hit["elevation"])
                    if hit.get("elevation") is not None
                    else None,
                    "population": int(hit["population"])
                    if hit.get("population")
                    else None,
                    "admin1": hit.get("admin1"),
                }
            )

    OUTPUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(out)} locations to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
