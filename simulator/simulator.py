"""
simulator/simulator.py
======================
Replays the 2018 data hour by hour to mimic a real-time feed.

For each simulated hour it:
  1. sends the actual reading to the backend's /predict endpoint,
  2. receives the next-24-hour forecast,
  3. writes the current state (recent actuals + latest forecast) to a JSON
     file that the Streamlit frontend reads to draw the live chart.

Before the loop it primes the backend with a 7-day warm-up window via /init.

Configuration (environment variables, all optional):
  BACKEND_URL    where the backend lives        (default http://127.0.0.1:8000)
  LIVE_DIR       folder for the live state file  (default live)
  SIM_DELAY      seconds to wait between hours    (default 1.0)
  SIM_MAX_STEPS  stop after N hours, 0 = all 2018 (default 0)

Run it from the PROJECT ROOT (with the backend already running) using:
    python simulator/simulator.py
Stop it any time with Ctrl+C.
"""

import os
import sys
import json
import time

sys.path.insert(0, "training")          # so we can import data_prep

import pandas as pd
import requests
from data_prep import build_simulation_data

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
LIVE_DIR = os.environ.get("LIVE_DIR", "live")
SIM_DELAY = float(os.environ.get("SIM_DELAY", "1.0"))
SIM_MAX_STEPS = int(os.environ.get("SIM_MAX_STEPS", "0"))   # 0 = run all of 2018

WINDOW = 168           # 7-day warm-up window
HISTORY_KEEP = 168     # how many recent hours to keep in the chart

os.makedirs(LIVE_DIR, exist_ok=True)
STATE_PATH = os.path.join(LIVE_DIR, "state.json")


def to_reading(timestamp, row):
    return {"timestamp": str(timestamp),
            "load": float(row["load"]),
            "temp": float(row["temp"]),
            "humidity": float(row["humidity"])}


def init_backend(payload):
    r = requests.post(f"{BACKEND_URL}/init", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def post_forecast(reading):
    r = requests.post(f"{BACKEND_URL}/predict", json=reading, timeout=30)
    r.raise_for_status()
    return r.json()


def write_state(now, history, forecast):
    """Write the live state atomically so the frontend never reads half a file."""
    state = {"now": str(now), "history": history, "forecast": forecast}
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def main():
    df = build_simulation_data()
    sim_start = pd.Timestamp("2018-01-01", tz="Europe/Madrid")
    warmup = df[df.index < sim_start].tail(WINDOW)
    stream = df[df.index >= sim_start]

    if len(warmup) < WINDOW:
        raise SystemExit(f"Not enough warm-up data: {len(warmup)}/{WINDOW}")

    # 1) prime the backend buffer with the last 7 days of 2017
    init_payload = {"history": [to_reading(ts, r) for ts, r in warmup.iterrows()]}
    print("init:", init_backend(init_payload))

    history = []            # list of {timestamp, actual, predicted}
    prev_forecast = None    # the 24h forecast produced in the previous step

    for i, (ts, row) in enumerate(stream.iterrows()):
        if SIM_MAX_STEPS and i >= SIM_MAX_STEPS:
            break

        actual = float(row["load"])

        # what did the model predict for THIS hour one step ago?
        predicted_now = None
        if prev_forecast and prev_forecast[0]["timestamp"] == str(ts):
            predicted_now = prev_forecast[0]["predicted"]

        history.append({"timestamp": str(ts),
                        "actual": round(actual, 1),
                        "predicted": predicted_now})
        history = history[-HISTORY_KEEP:]

        # 2) send the actual reading, get the next-24h forecast
        result = post_forecast(to_reading(ts, row))
        forecast = [{"timestamp": t, "predicted": v}
                    for t, v in zip(result["forecast_timestamps"],
                                    result["forecast_load_mw"])]
        prev_forecast = forecast

        # 3) publish the live state for the frontend
        write_state(ts, history, forecast)

        nxt = forecast[0]["predicted"]
        print(f"[{i + 1}] {ts}  actual={actual:,.0f} MW  next-hour forecast={nxt:,.0f} MW")
        time.sleep(SIM_DELAY)

    print("Simulation finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
