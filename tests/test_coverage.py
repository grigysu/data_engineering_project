"""Unit tests for the gold coverage / additive ingest logic.

No real S3 — boto3 is mocked. The pieces under test are pure:
  - partition-key parsing
  - manifest round-trip
  - missing-date math
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from ingestion.coverage import (
    MANIFEST_KEY,
    Coverage,
    daterange,
    missing_dates,
    read_manifest,
    scan_gold_coverage,
    write_manifest,
)


def test_daterange_inclusive():
    out = daterange(date(2026, 5, 1), date(2026, 5, 3))
    assert out == [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)]


def test_daterange_single_day():
    out = daterange(date(2026, 5, 1), date(2026, 5, 1))
    assert out == [date(2026, 5, 1)]


def test_daterange_inverted_returns_empty():
    assert daterange(date(2026, 5, 10), date(2026, 5, 1)) == []


def test_missing_dates_skips_present():
    present = {date(2026, 5, 2), date(2026, 5, 3)}
    out = missing_dates(date(2026, 5, 1), date(2026, 5, 4), present)
    assert out == [date(2026, 5, 1), date(2026, 5, 4)]


def test_missing_dates_all_present():
    present = {date(2026, 5, d) for d in range(1, 6)}
    assert missing_dates(date(2026, 5, 1), date(2026, 5, 5), present) == []


def test_coverage_json_roundtrip():
    cov = Coverage(
        archive={date(2026, 5, 1), date(2026, 5, 2)},
        forecast={date(2026, 5, 3)},
        scanned_at=datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc),
    )
    raw = cov.to_json()
    assert raw["archive"] == ["2026-05-01", "2026-05-02"]
    assert raw["forecast"] == ["2026-05-03"]
    restored = Coverage.from_json(raw)
    assert restored.archive == cov.archive
    assert restored.forecast == cov.forecast
    assert restored.scanned_at == cov.scanned_at


def _fake_paginator(keys: list[str]):
    """Return a list_objects_v2 paginator mock that yields one page of `keys`."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    return paginator


def test_scan_gold_coverage_extracts_partition_dates():
    s3 = MagicMock()
    s3.get_paginator.return_value = _fake_paginator(
        [
            "gold/weather_features/dataset=archive/date=2026-05-01/part-0.parquet",
            "gold/weather_features/dataset=archive/date=2026-05-02/part-0.parquet",
            "gold/weather_features/dataset=archive/date=2026-05-02/part-1.parquet",
            "gold/weather_features/dataset=archive/_SUCCESS",  # no date= -> ignored
        ]
    )
    # For both archive + forecast scans, the same paginator yields the same set;
    # what matters here is the parse logic, not which dataset.
    cov = scan_gold_coverage(s3, bucket="b")
    assert date(2026, 5, 1) in cov.archive
    assert date(2026, 5, 2) in cov.archive
    assert len(cov.archive) == 2


def test_write_then_read_manifest_roundtrip():
    s3 = MagicMock()
    # write_manifest captures the body
    captured = {}

    def fake_put(**kwargs):
        captured["Key"] = kwargs["Key"]
        captured["Body"] = kwargs["Body"]

    s3.put_object.side_effect = fake_put

    cov = Coverage(
        archive={date(2026, 5, 1)},
        scanned_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    write_manifest(s3, "weather-lake", cov)
    assert captured["Key"] == MANIFEST_KEY
    parsed = json.loads(captured["Body"])
    assert parsed["archive"] == ["2026-05-01"]

    # Now mock get_object to return what write_manifest produced.
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: captured["Body"])}
    loaded = read_manifest(s3, "weather-lake")
    assert loaded is not None
    assert loaded.archive == {date(2026, 5, 1)}


def test_read_manifest_returns_none_on_missing_key():
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    s3.get_object.side_effect = s3.exceptions.NoSuchKey
    assert read_manifest(s3, "weather-lake") is None
