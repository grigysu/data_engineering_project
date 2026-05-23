"""PyTorch Dataset over the Gold parquet.

Read the Spark-produced gold table from the local lake (MinIO via s3a is
overkill for the train loop — for small models we just sync the parquet
locally or read it via pyarrow), then window each per-location time
series into (seq_in, seq_out) pairs.

Time-based train/val split: windows are sorted by anchor timestamp, then
sliced — no random shuffling that would leak the future into the past.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import torch
from torch.utils.data import Dataset


# Inputs the model sees. We drop the categorical season and partition
# columns (dataset, date) — keep only numeric features.
FEATURE_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_2m_lag_1h",
    "temperature_2m_lag_3h",
    "temperature_2m_lag_24h",
    "temperature_2m_roll24_mean",
    "temperature_2m_roll24_std",
)
TARGET_COLUMN = "temperature_2m"


@dataclass(frozen=True)
class WindowSpec:
    seq_in: int = 24  # 24 hours of context
    seq_out: int = 6  # next 6 hours
    stride: int = 1


def load_gold(path: str | Path) -> pd.DataFrame:
    """Read the gold parquet (any partition layout) into a pandas DataFrame.

    Path can be a local directory, a single .parquet file, or anything
    pyarrow's Dataset abstraction can open.
    """
    df = pads.dataset(str(path), format="parquet").to_table().to_pandas()
    return df.sort_values(["lat", "lon", "observed_at"]).reset_index(drop=True)


def _build_windows(
    series: pd.DataFrame, spec: WindowSpec, feature_cols: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows over one location's time series.

    Returns (X, y, anchor_ts) of shapes:
        X:         (n_windows, seq_in,  n_features)
        y:         (n_windows, seq_out)
        anchor_ts: (n_windows,)   timestamp at the end of the input window
    """
    feats = series[list(feature_cols)].to_numpy(dtype=np.float32)
    target = series[TARGET_COLUMN].to_numpy(dtype=np.float32)
    ts = series["observed_at"].to_numpy()

    n = len(series)
    last_start = n - spec.seq_in - spec.seq_out
    if last_start < 0:
        return (
            np.empty((0, spec.seq_in, len(feature_cols)), dtype=np.float32),
            np.empty((0, spec.seq_out), dtype=np.float32),
            np.empty(0, dtype=ts.dtype),
        )
    starts = np.arange(0, last_start + 1, spec.stride)
    X = np.stack([feats[s : s + spec.seq_in] for s in starts])
    y = np.stack(
        [target[s + spec.seq_in : s + spec.seq_in + spec.seq_out] for s in starts]
    )
    anchors = np.array([ts[s + spec.seq_in - 1] for s in starts])
    return X, y, anchors


def build_window_set(
    gold: pd.DataFrame,
    spec: WindowSpec = WindowSpec(),
    feature_cols: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window all locations and concatenate. Drops rows with NaN in any feature."""
    clean = gold.dropna(subset=list(feature_cols) + [TARGET_COLUMN])
    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_a: list[np.ndarray] = []
    for _, series in clean.groupby(["lat", "lon"], sort=True):
        Xi, yi, ai = _build_windows(series, spec, feature_cols)
        if len(Xi) == 0:
            continue
        parts_X.append(Xi)
        parts_y.append(yi)
        parts_a.append(ai)
    if not parts_X:
        return (
            np.empty((0, spec.seq_in, len(feature_cols)), dtype=np.float32),
            np.empty((0, spec.seq_out), dtype=np.float32),
            np.empty(0),
        )
    return np.concatenate(parts_X), np.concatenate(parts_y), np.concatenate(parts_a)


def time_based_split(
    X: np.ndarray, y: np.ndarray, anchors: np.ndarray, val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by anchor timestamp: oldest `1-val_fraction` is train, newest is val."""
    order = np.argsort(anchors)
    X, y = X[order], y[order]
    cut = int(len(X) * (1 - val_fraction))
    return X[:cut], y[:cut], X[cut:], y[cut:]


class WeatherWindowsDataset(Dataset):
    """Minimal Dataset wrapping pre-built numpy arrays of windows.

    Normalization (StandardScaler-style) is applied lazily here so the
    same mean/std vector — fit on the training split only — can be
    reused for val/test. Targets are kept on their original scale
    (so MAE / RMSE on temperature are in degrees Celsius).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        if feature_mean is None or feature_std is None:
            feature_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            feature_std = X.reshape(-1, X.shape[-1]).std(axis=0)
            feature_std[feature_std == 0] = 1.0  # avoid /0 for constant cols
        self.X = ((X - feature_mean) / feature_std).astype(np.float32)
        self.y = y.astype(np.float32)
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
