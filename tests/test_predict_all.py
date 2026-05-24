"""Unit tests for predict_one_cell — pure-Python, no DB, no real PyTorch model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ml.predict_all import predict_one_cell


SEQ_IN = 4
SEQ_OUT = 3
FEATURE_COLUMNS = ("temperature_2m", "humidity")


def _stub_predictor(gold: pd.DataFrame, pred_values: list[float] | None = None):
    """Build a Predictor-shaped object whose .model returns fixed values.

    Returning a SimpleNamespace (not the real Predictor) avoids needing a
    checkpoint or MinIO connection in unit tests.
    """
    fixed = pred_values if pred_values is not None else [0.0] * SEQ_OUT

    class _StubModel:
        def __call__(self, x_t):
            return torch.tensor([fixed], dtype=torch.float32)

    return SimpleNamespace(
        gold=gold,
        feature_columns=FEATURE_COLUMNS,
        feature_mean=np.zeros(len(FEATURE_COLUMNS), dtype=np.float32),
        feature_std=np.ones(len(FEATURE_COLUMNS), dtype=np.float32),
        spec=SimpleNamespace(seq_in=SEQ_IN, seq_out=SEQ_OUT),
        device=torch.device("cpu"),
        model=_StubModel(),
        model_version="2026-05-25T00-00-00Z",
    )


def _gold_for(lat: float, lon: float, n_hours: int) -> pd.DataFrame:
    """Synthetic cell history. All feature columns present, no NaNs."""
    ts = pd.date_range("2026-05-20 00:00", periods=n_hours, freq="h")
    return pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "observed_at": ts,
            "temperature_2m": np.arange(n_hours, dtype=np.float32),
            "humidity": np.arange(n_hours, dtype=np.float32) + 50.0,
        }
    )


def test_predict_one_cell_emits_seq_out_rows():
    gold = _gold_for(40.0, 44.0, n_hours=10)
    p = _stub_predictor(gold, pred_values=[1.5, 2.5, 3.5])
    rows = predict_one_cell(p, 40.0, 44.0)
    assert rows is not None
    assert len(rows) == SEQ_OUT
    assert [r.predicted_value for r in rows] == [1.5, 2.5, 3.5]


def test_predict_one_cell_returns_none_when_history_below_seq_in():
    gold = _gold_for(40.0, 44.0, n_hours=SEQ_IN - 1)
    p = _stub_predictor(gold)
    assert predict_one_cell(p, 40.0, 44.0) is None


def test_predict_one_cell_returns_none_for_unknown_cell():
    gold = _gold_for(40.0, 44.0, n_hours=10)
    p = _stub_predictor(gold)
    # The cell exists in gold for (40, 44) but we ask for (41, 45) → empty slice.
    assert predict_one_cell(p, 41.0, 45.0) is None


def test_predict_one_cell_target_time_arithmetic():
    gold = _gold_for(40.0, 44.0, n_hours=10)
    p = _stub_predictor(gold)
    rows = predict_one_cell(p, 40.0, 44.0)
    anchor_naive = gold["observed_at"].iloc[-1]
    anchor_utc = anchor_naive.to_pydatetime().replace(tzinfo=timezone.utc)
    assert rows[0].target_time == anchor_utc + timedelta(hours=1)
    assert rows[1].target_time == anchor_utc + timedelta(hours=2)
    assert rows[2].target_time == anchor_utc + timedelta(hours=3)


def test_predict_one_cell_skips_nan_feature_rows():
    gold = _gold_for(40.0, 44.0, n_hours=SEQ_IN + 1)
    # Knock out a feature on a single row: cell falls below seq_in usable rows.
    gold.loc[2, "humidity"] = np.nan
    p = _stub_predictor(gold)
    # SEQ_IN + 1 = 5 rows, drop 1 → 4 usable, exactly enough for one anchor.
    rows = predict_one_cell(p, 40.0, 44.0)
    assert rows is not None and len(rows) == SEQ_OUT

    # But if we drop two rows, we're below seq_in → None.
    gold.loc[3, "humidity"] = np.nan
    p2 = _stub_predictor(gold)
    assert predict_one_cell(p2, 40.0, 44.0) is None


def test_predict_one_cell_uses_most_recent_anchor():
    """Anchor must be the LAST observed_at, not an arbitrary row."""
    gold = _gold_for(40.0, 44.0, n_hours=10)
    p = _stub_predictor(gold, pred_values=[7.0, 8.0, 9.0])
    rows = predict_one_cell(p, 40.0, 44.0)
    last_observed: pd.Timestamp = gold["observed_at"].iloc[-1]
    expected_first_target = last_observed.to_pydatetime().replace(
        tzinfo=timezone.utc
    ) + timedelta(hours=1)
    assert rows[0].target_time == expected_first_target
    assert isinstance(rows[0].target_time, datetime)
