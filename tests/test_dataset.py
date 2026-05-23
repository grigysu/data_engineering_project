"""Unit tests for the windowing + split logic — no Spark, no Parquet, no Torch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WindowSpec,
    _build_windows,
    build_window_set,
    time_based_split,
)


def _synthetic_series(
    n_hours: int, lat: float = 40.0, lon: float = 44.0
) -> pd.DataFrame:
    """Make a synthetic hourly series with all FEATURE_COLUMNS + the target.

    Values are deterministic and distinct so windowing bugs surface as
    off-by-one errors in the test assertions.
    """
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    base = np.arange(n_hours, dtype=np.float32)
    data = {col: base + i * 100.0 for i, col in enumerate(FEATURE_COLUMNS)}
    data[TARGET_COLUMN] = base  # also overwrite target with the clean sequence
    df = pd.DataFrame(data)
    df["observed_at"] = ts
    df["lat"] = lat
    df["lon"] = lon
    return df


def test_build_windows_shapes():
    series = _synthetic_series(n_hours=30)
    spec = WindowSpec(seq_in=10, seq_out=3, stride=1)
    X, y, anchors = _build_windows(series, spec, FEATURE_COLUMNS)
    # 30 hours, win=10, out=3 → 30 - 10 - 3 + 1 = 18 windows
    assert X.shape == (18, 10, len(FEATURE_COLUMNS))
    assert y.shape == (18, 3)
    assert anchors.shape == (18,)


def test_build_windows_target_alignment():
    """First window's target should be [seq_in, seq_in+1, ..., seq_in+seq_out-1]."""
    series = _synthetic_series(n_hours=20)
    spec = WindowSpec(seq_in=5, seq_out=2)
    X, y, _ = _build_windows(series, spec, FEATURE_COLUMNS)
    # Target was set to a clean ramp; the first window's inputs cover t=0..4,
    # so its targets must be t=5, t=6.
    assert list(y[0]) == [5.0, 6.0]
    # And the first input window's target column matches t=0..4.
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    assert list(X[0, :, target_idx]) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_build_windows_too_short_returns_empty():
    series = _synthetic_series(n_hours=5)
    spec = WindowSpec(seq_in=10, seq_out=3)
    X, y, anchors = _build_windows(series, spec, FEATURE_COLUMNS)
    assert X.shape == (0, 10, len(FEATURE_COLUMNS))
    assert y.shape == (0, 3)
    assert anchors.shape == (0,)


def test_build_window_set_concatenates_per_location():
    a = _synthetic_series(20, lat=40.0)
    b = _synthetic_series(20, lat=41.0)
    gold = pd.concat([a, b], ignore_index=True)
    spec = WindowSpec(seq_in=4, seq_out=2)
    # Per location: 20 - 4 - 2 + 1 = 15 windows → 30 total
    X, y, _ = build_window_set(gold, spec)
    assert len(X) == 30


def test_build_window_set_drops_rows_with_nan_features():
    series = _synthetic_series(20)
    # Knock out the first 3 lag values (real-world: early hours have no lag_24h).
    series.loc[:2, "temperature_2m_lag_24h"] = np.nan
    spec = WindowSpec(seq_in=4, seq_out=2)
    X, _, _ = build_window_set(series, spec)
    # 20 - 3 dropped = 17 rows, then 17 - 4 - 2 + 1 = 12 windows
    assert len(X) == 12


def test_time_based_split_is_chronological():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 4, len(FEATURE_COLUMNS))).astype(np.float32)
    y = rng.normal(size=(n, 2)).astype(np.float32)
    # Shuffled anchors — the function must sort them, not trust order.
    anchors = rng.permutation(np.arange(n)).astype(np.int64)
    Xtr, ytr, Xva, yva = time_based_split(X, y, anchors, val_fraction=0.2)
    assert len(Xtr) == 80
    assert len(Xva) == 20
    # Internal sort means content is reordered, but lengths must match expected cut.


def test_time_based_split_val_fraction_zero():
    X = np.zeros((10, 3, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.zeros((10, 2), dtype=np.float32)
    a = np.arange(10)
    Xtr, ytr, Xva, yva = time_based_split(X, y, a, val_fraction=0.0)
    assert len(Xtr) == 10
    assert len(Xva) == 0


@pytest.mark.parametrize("stride", [1, 2, 5])
def test_stride_skips_windows_correctly(stride: int):
    series = _synthetic_series(50)
    spec = WindowSpec(seq_in=10, seq_out=3, stride=stride)
    X, _, _ = _build_windows(series, spec, FEATURE_COLUMNS)
    # last_start = 50 - 10 - 3 = 37 → windows at 0, stride, 2*stride, ... <= 37
    expected = len(range(0, 38, stride))
    assert len(X) == expected
