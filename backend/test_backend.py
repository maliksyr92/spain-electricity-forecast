"""
backend/test_backend.py
=======================
A tiny client that checks your running FastAPI backend end-to-end,
WITHOUT needing the simulator yet.

It takes the last 7 days of real 2017 data as the "history", sends it to
/init, then sends one more hour to /predict and prints the 24h forecast.

Steps to use:
  1. In terminal #1 (project root):  uvicorn backend.main:app --port 8000
  2. In terminal #2 (project root):  python backend/test_backend.py
"""

import sys
sys.path.insert(0, "training")          # so we can import data_prep

import requests
from data_prep import build_dataset

BASE = "http://127.0.0.1:8000"
WINDOW = 168


def to_reading(timestamp, row):
    return {"timestamp": str(timestamp),
            "load": float(row["load"]),
            "temp": float(row["temp"]),
            "humidity": float(row["humidity"])}


def main():
    # health check
    print("health:", requests.get(f"{BASE}/health").json())

    # build a 7-day history + one extra hour from the end of 2017
    df = build_dataset()
    recent = df[df.index.year == 2017].tail(WINDOW + 1)
    history = recent.iloc[:WINDOW]
    new_hour = recent.iloc[WINDOW]

    # 1) initialize the buffer
    payload = {"history": [to_reading(ts, r) for ts, r in history.iterrows()]}
    print("init:", requests.post(f"{BASE}/init", json=payload).json())

    # 2) send the new actual reading and get the forecast
    reading = to_reading(new_hour.name, new_hour)
    resp = requests.post(f"{BASE}/predict", json=reading).json()

    print(f"\nLast actual demand: {reading['load']:,.0f} MW "
          f"at {reading['timestamp']}")
    print("Next 24h forecast (MW):")
    for t, v in zip(resp["forecast_timestamps"], resp["forecast_load_mw"]):
        print(f"  {t}  ->  {v:,.0f} MW")


if __name__ == "__main__":
    main()
