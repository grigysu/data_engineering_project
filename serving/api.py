"""FastAPI inference service for the weather LSTM.

Endpoints:
    GET  /health           - liveness + model/grid summary
    POST /forecast         - body {lat, lon} -> seq_out-hour temperature forecast
    GET  /grid_points      - list the lat/lon pairs we have data for

Configuration via env vars (also `.env`):
    MODEL_CHECKPOINT  default: checkpoints/best.pt
    GOLD_PATH         default: data/lake/gold/weather_features

Run:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from serving.predictor import NotEnoughHistory, Predictor

load_dotenv()


class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(os.getenv("MODEL_CHECKPOINT", "checkpoints/best.pt"))
    gold = Path(os.getenv("GOLD_PATH", "data/lake/gold/weather_features"))
    if not checkpoint.exists():
        raise RuntimeError(
            f"MODEL_CHECKPOINT not found at {checkpoint}. "
            "Run `python -m ml.train ...` first."
        )
    if not gold.exists():
        raise RuntimeError(
            f"GOLD_PATH not found at {gold}. "
            "Run the bronze->silver->gold Spark jobs first."
        )
    app.state.predictor = Predictor(checkpoint, gold)
    yield


app = FastAPI(
    title="Weather Forecast API",
    description="Multivariate-LSTM short-horizon temperature forecast over Armenia.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    predictor: Predictor = request.app.state.predictor
    return {
        "status": "ok",
        "device": str(predictor.device),
        "seq_in_hours": predictor.spec.seq_in,
        "horizon_hours": predictor.spec.seq_out,
        "feature_count": len(predictor.feature_columns),
        "grid_size": len(predictor._grid),
    }


@app.get("/grid_points")
def grid_points(request: Request) -> dict:
    predictor: Predictor = request.app.state.predictor
    return {
        "count": len(predictor._grid),
        "points": [{"lat": lat, "lon": lon} for lat, lon in predictor.grid_points()],
    }


@app.post("/forecast")
def forecast(req: ForecastRequest, request: Request) -> dict:
    predictor: Predictor = request.app.state.predictor
    try:
        response = predictor.predict(req.lat, req.lon)
    except NotEnoughHistory as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Convert dataclass tree to dict for FastAPI's default JSON encoder.
    return asdict(response)
