"""Seeded generator for the campus parcel-delay dataset.

Three versions share one schema but differ in distribution, so dataset
comparison and drift reasoning have real material:
  v1  baseline term-time operations
  v2  exam-week surge: heavier rider load, more evening traffic
  v3  monsoon: frequent rain/storm, longer detours
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ZONES = ["hostel", "library", "admin", "sports", "gate"]
PROFILES = {
    "v1": dict(n=6000, seed=101, rain=0.12, storm=0.03, load=(2.0, 1.0), evening=0.25, detour=1.0),
    "v2": dict(n=8000, seed=202, rain=0.12, storm=0.03, load=(3.4, 1.3), evening=0.45, detour=1.05),
    "v3": dict(n=9000, seed=303, rain=0.40, storm=0.15, load=(3.0, 1.2), evening=0.35, detour=1.35),
}


def generate(version: str) -> pd.DataFrame:
    p = PROFILES[version]
    rng = np.random.default_rng(p["seed"])
    n = p["n"]
    distance = np.clip(rng.gamma(2.2, 0.8, n) * p["detour"], 0.1, 9.0)
    evening = rng.random(n) < p["evening"]
    hour = np.where(evening, rng.integers(17, 22, n), rng.integers(8, 17, n))
    load = np.clip(rng.normal(*p["load"], n), 0, 9)
    draw = rng.random(n)
    weather = np.where(draw < p["storm"], "storm",
                       np.where(draw < p["storm"] + p["rain"], "rain", "clear"))
    zone = rng.choice(ZONES, n, p=[0.35, 0.2, 0.15, 0.1, 0.2])
    weight = np.clip(rng.lognormal(0.2, 0.7, n), 0.1, 25)
    logit = (-3.2 + 0.45 * distance + 0.55 * load + 0.6 * (weather == "rain")
             + 1.5 * (weather == "storm") + 0.5 * (hour >= 18)
             + 0.04 * weight + 0.3 * (zone == "hostel") + rng.normal(0, 0.4, n))
    delayed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(dict(distance_km=distance.round(2), hour=hour,
                             rider_load=load.round(2), weather=weather, zone=zone,
                             weight_kg=weight.round(2), delayed=delayed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=PROFILES, required=True)
    ap.add_argument("--out", default="data/parcels.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df = generate(a.version)
    df.to_csv(a.out, index=False)
    print(f"{a.version}: {len(df)} rows, delay rate {df.delayed.mean():.3f}")
