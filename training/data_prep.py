"""
data_prep.py
============
Loads the Kaggle Spain energy + weather data, cleans it, and builds one
tidy hourly table of model features for the offline period 2015-2017.

The 2018 part of the same files is reserved for the live simulator later,
so this module deliberately keeps only 2015-2017.
"""

import numpy as np
import pandas as pd

# Paths are relative to the project root (run scripts from the project root).
ENERGY_PATH = "data/energy_dataset.csv"
WEATHER_PATH = "data/weather_features.csv"

# The exact columns we feed into the model, in a FIXED order.
# The backend will rely on this same order later, so do not shuffle it.
FEATURE_COLUMNS = ["load", "temp", "humidity", "hour", "dayofweek", "month"]
TARGET_COLUMN = "load"


def load_energy(path: str = ENERGY_PATH) -> pd.DataFrame:
    """Read the energy file and return a clean hourly 'load' column (in MW)."""
    df = pd.read_csv(path)
    # Parse timestamps as UTC to avoid daylight-saving duplicate hours.
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    # We only need the national demand. Rename it to a short name: "load".
    load = df[["total load actual"]].rename(columns={"total load actual": "load"})

    # 0 MW (or negative) demand is physically impossible -> mark as missing.
    load.loc[load["load"] <= 0, "load"] = np.nan
    # Fill the few short gaps by interpolating over time.
    load["load"] = load["load"].interpolate(method="time").bfill().ffill()
    return load


def load_weather(path: str = WEATHER_PATH) -> pd.DataFrame:
    """Read the weather file and return national average temp (C) and humidity."""
    df = pd.read_csv(path)
    df["dt_iso"] = pd.to_datetime(df["dt_iso"], utc=True)

    # The file has one row per city per hour (5 Spanish cities).
    # Average temperature and humidity across cities to get one national value.
    agg = df.groupby("dt_iso")[["temp", "humidity"]].mean().sort_index()
    agg.index.name = "time"

    # Temperature is stored in Kelvin -> convert to Celsius (easier to read).
    agg["temp"] = agg["temp"] - 273.15

    agg = agg.interpolate(method="time").bfill().ffill()
    return agg


def build_dataset() -> pd.DataFrame:
    """Return one clean hourly DataFrame with all model features for 2015-2017."""
    load = load_energy()
    weather = load_weather()

    # Align demand and weather on the shared hourly timestamp.
    df = load.join(weather, how="inner")

    # Convert from UTC to local Spanish time so calendar features match the
    # real daily/weekly demand rhythm (people wake up at local 7am, not UTC).
    df.index = df.index.tz_convert("Europe/Madrid")

    # Calendar features.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek   # Monday = 0 ... Sunday = 6
    df["month"] = df.index.month

    # Keep only the offline period (2018 is for the live simulator later).
    df = df[(df.index.year >= 2015) & (df.index.year <= 2017)]

    df = df[FEATURE_COLUMNS].dropna()
    return df


if __name__ == "__main__":
    data = build_dataset()
    print("Rows:", len(data))
    print("Date range:", data.index.min(), "->", data.index.max())
    print(data.head())
    print(data.describe().round(1))
