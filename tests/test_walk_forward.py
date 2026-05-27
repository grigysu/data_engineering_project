"""Unit tests for walk-forward iteration helpers — no Torch model, no DB."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ml.walk_forward import (
    _to_utc_dt,
    build_inference_batch,
    enumerate_anchors,
    target_times_for,
)


# ---------- enumerate_anchors ----------


def test_enumerate_anchors_first_index_is_seq_in_minus_one():
    # 10 rows, seq_in=4 → first valid 'now' index is 3 (rows 0..3 = full window).
    assert enumerate_anchors(10, seq_in=4, stride_hours=1)[0] == 3


def test_enumerate_anchors_last_index_is_n_minus_one():
    # Final anchor is the most recent row; its forecast targets extend past data.
    out = enumerate_anchors(10, seq_in=4, stride_hours=1)
    assert out[-1] == 9


def test_enumerate_anchors_full_range_count():
    assert enumerate_anchors(10, seq_in=4, stride_hours=1) == [3, 4, 5, 6, 7, 8, 9]


def test_enumerate_anchors_too_short_returns_empty():
    assert enumerate_anchors(3, seq_in=4, stride_hours=1) == []


def test_enumerate_anchors_exactly_seq_in_rows_returns_one_anchor():
    assert enumerate_anchors(4, seq_in=4, stride_hours=1) == [3]


@pytest.mark.parametrize(
    "stride,expected",
    [
        (1, [3, 4, 5, 6, 7, 8, 9]),
        (2, [3, 5, 7, 9]),
        (3, [3, 6, 9]),
        (6, [3, 9]),
        # stride=0 must not infinite-loop: implementation clamps to 1.
        (0, [3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_enumerate_anchors_stride(stride: int, expected: list[int]):
    assert enumerate_anchors(10, seq_in=4, stride_hours=stride) == expected


# ---------- target_times_for ----------


def test_target_times_match_anchor_plus_k_hours():
    anchor = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    out = target_times_for(anchor, seq_out=3)
    assert out == [
        datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    ]


def test_target_times_seq_out_one():
    anchor = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    out = target_times_for(anchor, seq_out=1)
    assert out == [datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)]


# ---------- build_inference_batch ----------


def test_build_inference_batch_shapes_and_alignment():
    # 8 rows, 2 features; verify each anchor's window is the trailing seq_in rows.
    feats = np.arange(16, dtype=np.float32).reshape(8, 2)
    anchors = [3, 5, 7]
    batch = build_inference_batch(feats, anchors, seq_in=4)
    assert batch.shape == (3, 4, 2)
    # Window for anchor=3 must be rows 0..3 → values 0..7 reshape (4,2)
    np.testing.assert_array_equal(batch[0], feats[0:4])
    # Window for anchor=7 must be rows 4..7
    np.testing.assert_array_equal(batch[2], feats[4:8])


def test_build_inference_batch_single_anchor():
    feats = np.zeros((5, 3), dtype=np.float32)
    feats[2] = [1, 2, 3]
    batch = build_inference_batch(feats, [4], seq_in=5)
    assert batch.shape == (1, 5, 3)
    np.testing.assert_array_equal(batch[0], feats)


# ---------- _to_utc_dt ----------


def test_to_utc_dt_naive_becomes_utc():
    ts = pd.Timestamp("2026-05-01 12:00:00")  # naive
    out = _to_utc_dt(ts)
    assert out.tzinfo is timezone.utc
    assert out.hour == 12


def test_to_utc_dt_aware_preserved():
    ts = pd.Timestamp("2026-05-01 12:00:00", tz="UTC")
    out = _to_utc_dt(ts)
    assert out.tzinfo is not None
    assert out.hour == 12


def test_to_utc_dt_from_numpy_datetime64():
    ts = np.datetime64("2026-05-01T12:00:00")
    out = _to_utc_dt(ts)
    assert out.tzinfo is timezone.utc
    assert out.year == 2026 and out.month == 5 and out.day == 1
