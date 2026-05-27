"""Typed loader for `config/train.yaml` — the single source of truth for
ML hyperparameters.

Usage:
    from ml.config import load_config
    cfg = load_config()                 # default: <repo>/config/train.yaml
    cfg = load_config(Path("custom.yaml"))

    print(cfg.train.epochs, cfg.backtest.lookback_days)

CLI flags on `ml.train` / `ml.walk_forward` override these per-run; pass
`overrides={"train": {"epochs": 5}}` to `load_config` to apply them
programmatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train.yaml"


@dataclass(frozen=True)
class TrainConfig:
    seq_in: int = 48
    seq_out: int = 24
    hidden_size: int = 64
    num_layers: int = 2
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 42
    cutoff_days: int = 7


@dataclass(frozen=True)
class BacktestConfig:
    lookback_days: int = 7
    stride_hours: int = 24


@dataclass(frozen=True)
class PathsConfig:
    gold: str = "s3://weather-lake/gold/weather_features"
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class Config:
    train: TrainConfig
    backtest: BacktestConfig
    paths: PathsConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": asdict(self.train),
            "backtest": asdict(self.backtest),
            "paths": asdict(self.paths),
        }


def _filter_known(raw: dict[str, Any] | None, cls) -> dict[str, Any]:
    """Drop keys the dataclass doesn't recognize so YAML evolution doesn't
    break loading on older code. Logs nothing — surprises surface in tests."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (raw or {}).items() if k in known}


def load_config(
    path: Path | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> Config:
    """Load + validate the YAML config. Missing file → all-default Config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    if overrides:
        for section, kvs in overrides.items():
            if not kvs:
                continue
            raw.setdefault(section, {})
            raw[section].update(kvs)

    return Config(
        train=TrainConfig(**_filter_known(raw.get("train"), TrainConfig)),
        backtest=BacktestConfig(**_filter_known(raw.get("backtest"), BacktestConfig)),
        paths=PathsConfig(**_filter_known(raw.get("paths"), PathsConfig)),
    )
