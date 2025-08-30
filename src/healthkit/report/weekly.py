from __future__ import annotations

import pandas as pd


def weekly_mileage(workouts: pd.DataFrame) -> pd.DataFrame:
    if workouts.empty:
        return workouts
    df = workouts.copy()
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["week"] = df["start_utc"].dt.to_period("W-SUN").dt.start_time.dt.date
    df["distance_m"] = pd.to_numeric(
        df.get("gpx_distance_m", df.get("distance_m")), errors="coerce"
    )
    out = df.groupby("week")["distance_m"].sum().rename("distance_m").reset_index()
    out["distance_mi"] = out["distance_m"] / 1609.344
    return out
