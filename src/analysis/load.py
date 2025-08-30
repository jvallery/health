from __future__ import annotations

import numpy as np
import pandas as pd


def edwards_points(zone_minutes: dict[str, float]) -> float:
    w = {"z1_min": 1, "z2_min": 2, "z3_min": 3, "z4_min": 4, "z5_min": 5}
    return float(sum(zone_minutes.get(k, 0.0) * w[k] for k in w))


def trimp_banister(hr_series: pd.Series, hr_rest: float, hr_max: float, sex: str = "M") -> float:
    if hr_series is None or len(hr_series) == 0:
        return 0.0
    k = 1.92 if sex.upper().startswith("M") else 1.67
    dt_min = 1.0 / 60.0  # assume 1 Hz samples; otherwise resample
    ratio = (hr_series - hr_rest) / max(hr_max - hr_rest, 1.0)
    ratio = ratio.clip(lower=0)
    tr = float(np.sum(dt_min * ratio * np.exp(k * ratio)))
    return tr


def weekly_load(workouts: pd.DataFrame) -> pd.DataFrame:
    if workouts.empty:
        return pd.DataFrame()
    w = workouts.copy()
    w["start_utc"] = pd.to_datetime(w.get("start_utc", w.get("start_local")), utc=True)
    if "trimp" not in w.columns:
        # Fallback: use Edwards points if present
        w["trimp"] = w.get("edwards_points", pd.Series(0.0, index=w.index)).fillna(0.0)
    w["week"] = w["start_utc"].dt.to_period("W-MON").dt.start_time
    g = w.groupby("week")["trimp"].sum().rename("weekly_trimp").reset_index()
    return g


def acute_chronic_ratio(weekly_load_df: pd.DataFrame) -> pd.DataFrame:
    if weekly_load_df.empty:
        return weekly_load_df
    d = weekly_load_df.copy()
    d = d.sort_values("week").reset_index(drop=True)
    d["acute_7d"] = d["weekly_trimp"].rolling(1).sum()
    d["chronic_28d"] = d["weekly_trimp"].rolling(4).mean()
    d["ac_ratio"] = d["acute_7d"] / d["chronic_28d"]
    return d
