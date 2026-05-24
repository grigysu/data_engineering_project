"""Predictor: load a trained checkpoint + the gold feature store, then produce
a `seq_out`-hour temperature forecast for an arbitrary (lat, lon).

Design choices:
  - Checkpoint + gold parquet are loaded once at construction; subsequent
    `.predict()` calls reuse them. FastAPI's lifespan hook gets one
    Predictor per app instance.
  - Arbitrary (lat, lon) requests are snapped to the nearest grid cell we
    actually have data for (Open-Meteo answers on the ERA5 0.25° grid).
  - Features come from the gold parquet rather than being re-derived at
    request time — that guarantees the inference-time schema matches
    training exactly. Cost: the lake must be re-loaded for fresh data
    (out of scope here; document that the server needs a restart, or add
    a /reload endpoint later).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ml.dataset import WindowSpec, load_gold
from ml.models.lstm import WeatherLSTM


@dataclass(frozen=True)
class ForecastPrediction:
    hours_ahead: int
    temperature_2m_c: float


@dataclass(frozen=True)
class ForecastResponse:
    requested_lat: float
    requested_lon: float
    snapped_lat: float
    snapped_lon: float
    forecast_anchor: str  # ISO timestamp of the most recent observation used as context
    seq_in_hours: int
    horizon_hours: int
    predictions: list[ForecastPrediction]


class NotEnoughHistory(ValueError):
    """The grid point has fewer than seq_in usable rows (NaN-feature drops)."""


class Predictor:
    def __init__(
        self,
        checkpoint_path: Path | str,
        gold_path: Path | str,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_columns: tuple[str, ...] = tuple(ckpt["feature_columns"])
        self.spec = WindowSpec(**ckpt["spec"])
        self.feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        hp = ckpt["hyperparams"]

        self.model = WeatherLSTM(
            n_features=len(self.feature_columns),
            seq_out=self.spec.seq_out,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.gold = load_gold(gold_path)
        self._grid = (
            self.gold[["lat", "lon"]].drop_duplicates().to_numpy(dtype=np.float64)
        )

    def grid_points(self) -> list[tuple[float, float]]:
        return [(float(lat), float(lon)) for lat, lon in self._grid]

    def _nearest_grid_point(self, lat: float, lon: float) -> tuple[float, float]:
        # Plain Euclidean over (lat, lon) — fine for the small spatial extent
        # we cover (Armenia). A real system over larger areas would use
        # haversine + a KD-tree.
        d2 = (self._grid[:, 0] - lat) ** 2 + (self._grid[:, 1] - lon) ** 2
        i = int(np.argmin(d2))
        return float(self._grid[i, 0]), float(self._grid[i, 1])

    def predict(self, lat: float, lon: float) -> ForecastResponse:
        snapped_lat, snapped_lon = self._nearest_grid_point(lat, lon)

        rows = self.gold[
            (self.gold["lat"] == snapped_lat) & (self.gold["lon"] == snapped_lon)
        ].sort_values("observed_at")
        rows = rows.dropna(subset=list(self.feature_columns))
        if len(rows) < self.spec.seq_in:
            raise NotEnoughHistory(
                f"grid point ({snapped_lat}, {snapped_lon}) has {len(rows)} usable "
                f"rows; need {self.spec.seq_in} for one inference window."
            )

        window = rows.iloc[-self.spec.seq_in :]
        x = window[list(self.feature_columns)].to_numpy(dtype=np.float32)
        x = (x - self.feature_mean) / self.feature_std

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            pred = self.model(x_t).cpu().numpy().squeeze(0)

        anchor_ts = window["observed_at"].iloc[-1]
        return ForecastResponse(
            requested_lat=float(lat),
            requested_lon=float(lon),
            snapped_lat=snapped_lat,
            snapped_lon=snapped_lon,
            forecast_anchor=str(anchor_ts),
            seq_in_hours=self.spec.seq_in,
            horizon_hours=self.spec.seq_out,
            predictions=[
                ForecastPrediction(hours_ahead=i + 1, temperature_2m_c=float(pred[i]))
                for i in range(self.spec.seq_out)
            ],
        )
