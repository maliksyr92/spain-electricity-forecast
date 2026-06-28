"""
backend/main.py
===============
A small FastAPI service that serves 24-hour-ahead forecasts of Spain's
national electricity demand using the LSTM model trained in Phase 1.

How it works
------------
* On startup it loads the saved model + scalers + feature config from models/.
* It keeps a rolling "buffer" of the most recent WINDOW (168) hourly readings.
* POST /init      -> fill the buffer with a 7-day history (168 readings).
* POST /predict   -> add ONE new hourly reading, then return the next 24h forecast.
* GET  /health    -> simple status check.

Run it locally from the PROJECT ROOT with:
    uvicorn backend.main:app --reload --port 8000
Then open the interactive docs at:  http://127.0.0.1:8000/docs
"""

import os
import json
from collections import deque
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Folder that contains the artifacts produced in Phase 1.
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# The exact feature order the model was trained with. MUST match data_prep.py.
EXPECTED_FEATURES = ["load", "temp", "humidity", "hour", "dayofweek", "month"]

# Holds the loaded artifacts and rolling state (single-process container).
ML = {"model": None, "feature_scaler": None, "target_scaler": None,
      "window": 168, "horizon": 24}
STATE = {"buffer": deque(maxlen=168), "last_ts": None}


# ---------------------------------------------------------------------------
# Loading the trained artifacts
# ---------------------------------------------------------------------------
def load_artifacts(model_dir: str = MODEL_DIR) -> None:
    import tensorflow as tf  # imported lazily so the file imports even without TF

    with open(os.path.join(model_dir, "feature_config.json")) as f:
        cfg = json.load(f)

    if cfg["feature_columns"] != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Feature order mismatch. Model expects {cfg['feature_columns']}, "
            f"backend builds {EXPECTED_FEATURES}. Retrain or fix the order.")

    ML["window"] = cfg["window"]
    ML["horizon"] = cfg["horizon"]
    ML["model"] = tf.keras.models.load_model(
        os.path.join(model_dir, "lstm_model.keras"))
    ML["feature_scaler"] = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
    ML["target_scaler"] = joblib.load(os.path.join(model_dir, "target_scaler.pkl"))
    STATE["buffer"] = deque(maxlen=ML["window"])
    STATE["last_ts"] = None
    print(f"Artifacts loaded. window={ML['window']} horizon={ML['horizon']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()          # runs once when the server starts
    yield


app = FastAPI(title="Spain Electricity Demand Forecast API", lifespan=lifespan)
# Allow the Streamlit frontend (any origin) to call this API.
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------
class Reading(BaseModel):
    timestamp: str   # ISO time, local Spanish time, e.g. "2017-12-31 23:00:00+01:00"
    load: float      # actual demand at that hour, in MW
    temp: float      # national average temperature, in Celsius
    humidity: float  # national average humidity, in %


class InitRequest(BaseModel):
    history: List[Reading]   # should contain WINDOW (168) readings


class ForecastResponse(BaseModel):
    last_timestamp: str
    forecast_timestamps: List[str]
    forecast_load_mw: List[float]


# ---------------------------------------------------------------------------
# Pure helper logic (no FastAPI here -> easy to test)
# ---------------------------------------------------------------------------
def reading_to_row(timestamp: str, load: float, temp: float, humidity: float):
    """Turn one reading into a feature row in the EXPECTED_FEATURES order."""
    ts = pd.to_datetime(timestamp)
    row = [float(load), float(temp), float(humidity),
           float(ts.hour), float(ts.dayofweek), float(ts.month)]
    return row, ts


def compute_forecast(buffer_rows, last_ts, model, feature_scaler,
                     target_scaler, window, horizon):
    """Scale the buffer, run the model, and return (future_timestamps, mw_values)."""
    if len(buffer_rows) < window:
        raise ValueError(f"buffer has {len(buffer_rows)}/{window} rows")
    arr = np.asarray(buffer_rows, dtype="float32")          # (window, n_features)
    scaled = feature_scaler.transform(arr)
    X = scaled.reshape(1, window, arr.shape[1])
    pred_scaled = model.predict(X, verbose=0)               # (1, horizon)
    pred_mw = target_scaler.inverse_transform(pred_scaled)[0]   # (horizon,)
    future = pd.date_range(last_ts + pd.Timedelta(hours=1),
                           periods=horizon, freq="h")
    return future, pred_mw


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok",
            "model_loaded": ML["model"] is not None,
            "buffer_size": len(STATE["buffer"]),
            "window": ML["window"]}


@app.post("/init")
def init(req: InitRequest):
    STATE["buffer"].clear()
    last_ts = None
    for r in req.history:
        row, ts = reading_to_row(r.timestamp, r.load, r.temp, r.humidity)
        STATE["buffer"].append(row)
        last_ts = ts
    STATE["last_ts"] = last_ts
    return {"status": "initialized",
            "buffer_size": len(STATE["buffer"]),
            "window": ML["window"],
            "ready": len(STATE["buffer"]) >= ML["window"]}


@app.post("/predict", response_model=ForecastResponse)
def predict(reading: Reading):
    row, ts = reading_to_row(reading.timestamp, reading.load,
                             reading.temp, reading.humidity)
    STATE["buffer"].append(row)        # deque automatically drops the oldest
    STATE["last_ts"] = ts
    try:
        future, pred_mw = compute_forecast(
            list(STATE["buffer"]), STATE["last_ts"], ML["model"],
            ML["feature_scaler"], ML["target_scaler"],
            ML["window"], ML["horizon"])
    except ValueError as e:
        raise HTTPException(status_code=400,
            detail=f"{e}. Call /init with {ML['window']} readings first.")
    return ForecastResponse(
        last_timestamp=str(ts),
        forecast_timestamps=[str(t) for t in future],
        forecast_load_mw=[round(float(v), 1) for v in pred_mw],
    )
