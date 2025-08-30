from __future__ import annotations

import numpy as np
import pandas as pd


def zones_for_workout(
    hr_samples: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, hrmax: float
) -> dict:
    """Compute simple zone minutes (percent of HRmax)."""
    window = hr_samples[
        (hr_samples["timestamp_utc"] >= start) & (hr_samples["timestamp_utc"] <= end)
    ]
    if window.empty:
        return {k: 0.0 for k in ["z1_min", "z2_min", "z3_min", "z4_min", "z5_min"]}
    bins = [0, 0.6 * hrmax, 0.7 * hrmax, 0.8 * hrmax, 0.9 * hrmax, 10 * hrmax]
    cats = pd.cut(window["bpm"], bins=bins, labels=["z1", "z2", "z3", "z4", "z5"], right=False)
    counts = cats.value_counts()
    minutes = (counts / 60.0).to_dict()
    return {f"{k}_min": float(minutes.get(k, 0.0)) for k in ["z1", "z2", "z3", "z4", "z5"]}


def edwards_points_from_minutes(zmins: dict) -> float:
    weights = {"z1_min": 1, "z2_min": 2, "z3_min": 3, "z4_min": 4, "z5_min": 5}
    return float(sum(zmins.get(k, 0.0) * w for k, w in weights.items()))


def estimate_hrmax(workouts: pd.DataFrame | None, hr_samples: pd.DataFrame | None) -> float | None:
    vals = []
    if workouts is not None and not workouts.empty and "max_hr_bpm" in workouts:
        v = pd.to_numeric(workouts["max_hr_bpm"], errors="coerce").dropna()
        if not v.empty:
            vals.append(float(v.max()))
    if hr_samples is not None and not hr_samples.empty:
        v = pd.to_numeric(hr_samples["bpm"], errors="coerce").dropna()
        if not v.empty:
            vals.append(float(np.percentile(v, 95)))
    if not vals:
        return None
    return float(max(vals))


def zones_and_load_for_all(
    workouts: pd.DataFrame, hr_samples: pd.DataFrame, hrmax: float
) -> pd.DataFrame:
    if workouts.empty or hr_samples.empty or not hrmax:
        return pd.DataFrame(
            columns=[
                "workout_id",
                "z1_min",
                "z2_min",
                "z3_min",
                "z4_min",
                "z5_min",
                "edwards_points",
                "drop_1min_bpm",
                "drop_2min_bpm",
            ]
        )
    w = workouts.copy()
    w["start_utc"] = pd.to_datetime(w["start_utc"], utc=True)
    w["end_utc"] = pd.to_datetime(w["end_utc"], utc=True)
    rows: list[dict] = []
    for _, row in w.iterrows():
        z = zones_for_workout(hr_samples, row["start_utc"], row["end_utc"], hrmax)
        # Recovery drops
        pre = hr_samples[
            (hr_samples["timestamp_utc"] > row["end_utc"] - pd.Timedelta(seconds=60))
            & (hr_samples["timestamp_utc"] <= row["end_utc"])
        ]
        post1 = hr_samples[
            (hr_samples["timestamp_utc"] > row["end_utc"])
            & (hr_samples["timestamp_utc"] <= row["end_utc"] + pd.Timedelta(seconds=60))
        ]
        post2 = hr_samples[
            (hr_samples["timestamp_utc"] > row["end_utc"] + pd.Timedelta(seconds=60))
            & (hr_samples["timestamp_utc"] <= row["end_utc"] + pd.Timedelta(seconds=120))
        ]
        pre_mean = float(pre["bpm"].mean()) if not pre.empty else np.nan
        post1_mean = float(post1["bpm"].mean()) if not post1.empty else np.nan
        post2_mean = float(post2["bpm"].mean()) if not post2.empty else np.nan
        drop1 = (
            float(pre_mean - post1_mean)
            if not np.isnan(pre_mean) and not np.isnan(post1_mean)
            else np.nan
        )
        drop2 = (
            float(pre_mean - post2_mean)
            if not np.isnan(pre_mean) and not np.isnan(post2_mean)
            else np.nan
        )
        rows.append(
            {
                "workout_id": row["workout_id"],
                **z,
                "edwards_points": edwards_points_from_minutes(z),
                "drop_1min_bpm": drop1,
                "drop_2min_bpm": drop2,
            }
        )
    return pd.DataFrame(rows)
