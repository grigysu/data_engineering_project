"""Unit tests for the healthcheck module — no real network calls."""

from __future__ import annotations

from ingestion.healthcheck import (
    DEGRADED,
    DOWN,
    OK,
    EndpointHealth,
    ProbeResult,
    render_json,
    render_text,
)


def _ok(name: str) -> ProbeResult:
    return ProbeResult(name=name, success=True, detail="ok", elapsed_ms=10.0)


def _fail(name: str, detail: str = "boom") -> ProbeResult:
    return ProbeResult(name=name, success=False, detail=detail, elapsed_ms=5.0)


def test_render_text_marks_success_and_failure():
    health = EndpointHealth(name="archive", url="https://x.example/v1", verdict=DOWN)
    health.add(_ok("dns"))
    health.add(_fail("http_query", "HTTP 504 Gateway Time-out"))
    out = render_text([health])
    assert "[DOWN" in out
    assert "OK dns" in out
    assert "!! http_query" in out
    assert "504" in out


def test_render_json_round_trips_structure():
    import json

    health = EndpointHealth(name="forecast", url="https://y/v1", verdict=OK)
    health.add(_ok("dns"))
    health.add(_ok("http_query"))
    parsed = json.loads(render_json([health]))
    assert len(parsed) == 1
    assert parsed[0]["name"] == "forecast"
    assert parsed[0]["verdict"] == OK
    assert len(parsed[0]["probes"]) == 2
    assert parsed[0]["probes"][0]["success"] is True


def test_verdict_constants_are_distinct():
    assert {OK, DEGRADED, DOWN} == {"OK", "DEGRADED", "DOWN"}
