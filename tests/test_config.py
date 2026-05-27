"""Unit tests for ml.config — YAML load + CLI overrides + cutoff trim."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.config import (
    BacktestConfig,
    Config,
    PathsConfig,
    TrainConfig,
    load_config,
)
from ml.train import trim_to_cutoff


# ---------- load_config ----------


def test_load_config_missing_file_returns_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert isinstance(cfg, Config)
    assert cfg.train == TrainConfig()
    assert cfg.backtest == BacktestConfig()
    assert cfg.paths == PathsConfig()


def test_load_config_reads_yaml(tmp_path: Path):
    p = tmp_path / "train.yaml"
    p.write_text(
        "train:\n  epochs: 7\n  cutoff_days: 14\nbacktest:\n  stride_hours: 6\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.train.epochs == 7
    assert cfg.train.cutoff_days == 14
    # Sections not in YAML keep their dataclass defaults.
    assert cfg.train.seq_in == TrainConfig.seq_in
    assert cfg.backtest.stride_hours == 6
    assert cfg.backtest.lookback_days == BacktestConfig.lookback_days


def test_load_config_unknown_keys_dropped(tmp_path: Path):
    """Forward-compat: future YAML keys don't crash older code that loads them."""
    p = tmp_path / "train.yaml"
    p.write_text(
        "train:\n  epochs: 3\n  future_flag: true\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.train.epochs == 3  # known key honored
    # `future_flag` silently ignored — no AttributeError.


def test_load_config_overrides_take_precedence(tmp_path: Path):
    p = tmp_path / "train.yaml"
    p.write_text("train:\n  epochs: 5\n", encoding="utf-8")
    cfg = load_config(p, overrides={"train": {"epochs": 99}})
    assert cfg.train.epochs == 99


def test_load_config_empty_overrides_section_is_noop(tmp_path: Path):
    p = tmp_path / "train.yaml"
    p.write_text("train:\n  epochs: 5\n", encoding="utf-8")
    cfg = load_config(p, overrides={"train": {}})
    assert cfg.train.epochs == 5


def test_to_dict_roundtrip_keys():
    cfg = load_config(Path("/does/not/exist.yaml"))  # defaults
    d = cfg.to_dict()
    assert set(d) == {"train", "backtest", "paths"}
    assert d["train"]["epochs"] == TrainConfig.epochs


# ---------- trim_to_cutoff ----------


def _gold(start: str, n_hours: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({"observed_at": ts, "value": range(n_hours)})


def test_trim_to_cutoff_zero_is_noop():
    g = _gold("2026-05-20 00:00", 24)
    out = trim_to_cutoff(g, cutoff_days=0)
    assert len(out) == len(g)


def test_trim_to_cutoff_drops_last_n_days():
    # 10 days of hourly data → 240 rows. cutoff=2 → keep first 8 days.
    g = _gold("2026-05-20 00:00", 24 * 10)
    out = trim_to_cutoff(g, cutoff_days=2)
    # T = last timestamp = start + 239h. cutoff = T - 48h.
    # Rows whose observed_at <= cutoff: indices 0..191 (192 rows).
    assert len(out) == 192
    assert out["observed_at"].max() == g["observed_at"].iloc[191]


def test_trim_to_cutoff_empty_input():
    g = pd.DataFrame({"observed_at": pd.to_datetime([]), "value": []})
    out = trim_to_cutoff(g, cutoff_days=7)
    assert out.empty


@pytest.mark.parametrize("n_days, cutoff_days", [(1, 7), (3, 7)])
def test_trim_to_cutoff_larger_than_window_empties(n_days: int, cutoff_days: int):
    g = _gold("2026-05-20 00:00", 24 * n_days)
    out = trim_to_cutoff(g, cutoff_days=cutoff_days)
    # cutoff is before the dataset's earliest timestamp → no rows kept.
    assert out.empty
