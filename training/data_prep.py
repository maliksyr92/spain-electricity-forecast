"""
data_prep.py  (updated in Phase 3 — fully backward compatible)
==============================================================
Loads the Kaggle Spain energy + weather data, cleans it, and builds tidy
hourly feature tables.

* build_dataset()            -> 2015-2017  (used for training, UNCHANGED)
* build_simulation_data()    -> 2017+2018  (NEW: used by the simulator)

Nothing about the training behaviour changed; build_dataset() returns
exactly the same rows as before. We only added a second function.
"""

import numpy as np
import pandas as pd

ENERGY_PATH = "data/energy_dataset.csv"
WEATHER_PATH = "data/weather_features.csv"

# The exact columns we feed into the model, in a FIXED order.
FEATURE_COLUMNS = ["load", "temp", "humidity", "hour", "dayofweek", "month"]
TARGET_COLUMN = "load"


def load_energy(path: str = ENERGY_PATH) -> pd.DataFrame:
    """Read the energy file and return a clean hourly 'load' column (in MW)."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    load = df[["total load actual"]].rename(columns={"total load actual": "load"})
    load.loc[load["load"] <= 0, "load"] = np.nan           # 0 MW is impossible
    load["load"] = load["load"].interpolate(method="time").bfill().ffill()
    return load


def load_weather(path: str = WEATHER_PATH) -> pd.DataFrame:
    """Read the weather file and return national average temp (C) and humidity."""
    df = pd.read_csv(path)
    df["dt_iso"] = pd.to_datetime(df["dt_iso"], utc=True)
    agg = df.groupby("dt_iso")[["temp", "humidity"]].mean().sort_index()
    agg.index.name = "time"
    agg["temp"] = agg["temp"] - 273.15                     # Kelvin -> Celsius
    agg = agg.interpolate(method="time").bfill().ffill()
    return agg


def _assemble() -> pd.DataFrame:
    """Join demand + weather, convert to local time, add calendar features."""
    load = load_energy()
    weather = load_weather()
    df = load.join(weather, how="inner")
    # local Spanish time so the daily/weekly rhythm is correct
    df.index = df.index.tz_convert("Europe/Madrid")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def build_dataset() -> pd.DataFrame:
    """Clean hourly features for 2015-2017 (used for training). UNCHANGED."""
    df = _assemble()
    df = df[(df.index.year >= 2015) & (df.index.year <= 2017)]
    return df[FEATURE_COLUMNS].dropna()


def build_simulation_data() -> pd.DataFrame:
    """Clean hourly features for 2017+2018 (used by the live simulator).

    2017 is included so the simulator has a 7-day warm-up window right
    before the first hour of 2018.
    """
    df = _assemble()
    df = df[df.index.year >= 2017]
    return df[FEATURE_COLUMNS].dropna()


if __name__ == "__main__":
    train = build_dataset()
    sim = build_simulation_data()
    print("training rows (2015-2017):", len(train),
          "| range:", train.index.min(), "->", train.index.max())
    print("simulation rows (2017-2018):", len(sim),
          "| range:", sim.index.min(), "->", sim.index.max())
