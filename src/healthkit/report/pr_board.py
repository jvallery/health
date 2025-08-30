from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def fastest_by_bucket(
    workouts: pd.DataFrame, buckets_mi: Iterable[float] | None = None
) -> pd.DataFrame:
    if buckets_mi is None:
        buckets_mi = [1, 3.1, 5, 10]
    if workouts.empty:
        return workouts
    df = workouts.copy()
    df["distance_mi"] = (
        pd.to_numeric(df.get("gpx_distance_m", df.get("distance_m")), errors="coerce") / 1609.344
    )
    df = df.dropna(subset=["pace_min_per_mile", "distance_mi"])  # type: ignore[arg-type]
    rows = []
    for b in buckets_mi:
        subset = df[(df["distance_mi"] >= 0.9 * b) & (df["distance_mi"] <= 1.1 * b)]
        if subset.empty:
            continue
        best = subset.sort_values("pace_min_per_mile").iloc[0]
        rows.append(
            {
                "bucket_mi": b,
                "workout_id": best["workout_id"],
                "pace_min_per_mile": best["pace_min_per_mile"],
            }
        )
    return pd.DataFrame(rows)
