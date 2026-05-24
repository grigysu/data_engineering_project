"""Gold-layer coverage tracking for additive ingest.

The Spark gold job partitions by `(dataset, date)`, writing all 100 grid
cells together. So a partition's existence implies coverage for every
cell on that day — partition-level granularity is enough.

Coverage is the set of dates present in gold for a given dataset (e.g.
'archive'). We scan it once (cheap, just list keys) and cache the result
to `s3://<bucket>/manifest/ingested.json` so subsequent runs skip the
scan. Ingest reads the manifest, diffs against the requested range, and
only fetches missing dates.

Gold is the source of truth: if a date isn't in gold (e.g. silver/gold
processing failed midway), we'll re-ingest + re-process. Intentional.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

MANIFEST_KEY = "manifest/ingested.json"
DEFAULT_GOLD_PREFIX = "gold/weather_features"

_DATE_PARTITION_RE = re.compile(r"/date=(\d{4}-\d{2}-\d{2})/")


@dataclass
class Coverage:
    """Set of dates present in gold, per dataset partition."""

    archive: set[date] = field(default_factory=set)
    forecast: set[date] = field(default_factory=set)
    scanned_at: datetime | None = None

    def to_json(self) -> dict:
        return {
            "scanned_at": (self.scanned_at or datetime.now(timezone.utc)).isoformat(),
            "archive": sorted(d.isoformat() for d in self.archive),
            "forecast": sorted(d.isoformat() for d in self.forecast),
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Coverage":
        return cls(
            archive={date.fromisoformat(d) for d in raw.get("archive", [])},
            forecast={date.fromisoformat(d) for d in raw.get("forecast", [])},
            scanned_at=(
                datetime.fromisoformat(raw["scanned_at"])
                if raw.get("scanned_at")
                else None
            ),
        )

    def dataset_dates(self, dataset: str) -> set[date]:
        if dataset == "archive":
            return self.archive
        if dataset == "forecast":
            return self.forecast
        raise ValueError(f"unknown dataset: {dataset!r}")


def scan_gold_coverage(
    s3_client,
    bucket: str,
    prefix: str = DEFAULT_GOLD_PREFIX,
) -> Coverage:
    """List gold partition keys and extract the date set for each dataset.

    Reads only object *keys* (no parquet content), so this is O(num partitions)
    on metadata. Safe to call on every ingest run, but the manifest cache
    avoids even that.
    """
    cov = Coverage(scanned_at=datetime.now(timezone.utc))
    for ds in ("archive", "forecast"):
        ds_prefix = f"{prefix}/dataset={ds}/"
        cov.dataset_dates(ds).update(_list_dates_under(s3_client, bucket, ds_prefix))
    return cov


def _list_dates_under(s3_client, bucket: str, prefix: str) -> set[date]:
    """List all keys under `prefix` and pull out `date=YYYY-MM-DD` partitions."""
    dates: set[date] = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            m = _DATE_PARTITION_RE.search("/" + obj["Key"])
            if m:
                try:
                    dates.add(date.fromisoformat(m.group(1)))
                except ValueError:
                    pass
    return dates


def read_manifest(s3_client, bucket: str) -> Coverage | None:
    """Load the manifest from `s3://<bucket>/manifest/ingested.json`, or None."""
    try:
        body = s3_client.get_object(Bucket=bucket, Key=MANIFEST_KEY)["Body"].read()
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        # boto3 may raise a generic ClientError for 404; check the code.
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in (
            "NoSuchKey",
            "404",
        ):
            return None
        raise
    return Coverage.from_json(json.loads(body))


def write_manifest(s3_client, bucket: str, coverage: Coverage) -> None:
    body = json.dumps(coverage.to_json(), indent=2, sort_keys=True).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=MANIFEST_KEY,
        Body=body,
        ContentType="application/json",
    )


def load_or_scan_coverage(s3_client, bucket: str) -> Coverage:
    """Read the manifest if it exists; otherwise scan gold + seed the manifest."""
    cached = read_manifest(s3_client, bucket)
    if cached is not None:
        return cached
    fresh = scan_gold_coverage(s3_client, bucket)
    # Best-effort seed; ignore put failures (caller can still proceed).
    try:
        write_manifest(s3_client, bucket, fresh)
    except Exception:
        pass
    return fresh


def daterange(start: date, end: date) -> list[date]:
    """Inclusive list of dates in [start, end]."""
    n_days = (end - start).days + 1
    if n_days <= 0:
        return []
    return [date.fromordinal(start.toordinal() + i) for i in range(n_days)]


def missing_dates(start: date, end: date, present: set[date]) -> list[date]:
    """Dates in [start, end] not in `present`, in ascending order."""
    return [d for d in daterange(start, end) if d not in present]


def s3_client_from_env():
    """boto3 S3 client wired to MinIO via env vars. Used by ingest + coverage."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ROOT_USER"],
        aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
        region_name=os.getenv("MINIO_REGION", "us-east-1"),
    )
