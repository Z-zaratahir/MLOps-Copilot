"""DVC pipeline stage: profile the tracked dataset into a small metrics file."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "parcels.csv")
    profile = {"rows": len(df), "delay_rate": round(float(df["delayed"].mean()), 4),
               "mean_distance_km": round(float(df["distance_km"].mean()), 3),
               "mean_rider_load": round(float(df["rider_load"].mean()), 3),
               "storm_or_rain_share": round(float(df["weather"].isin(["rain", "storm"]).mean()), 4),
               "null_cells": int(df.isnull().sum().sum())}
    out = ROOT / "metrics" / "data_profile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(profile, indent=2) + "\n")
    print(profile)


if __name__ == "__main__":
    main()
