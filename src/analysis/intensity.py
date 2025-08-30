from __future__ import annotations

import numpy as np
import pandas as pd

from .common import add_effective_fields
from .load import trimp_banister


def estimate_hrmax(wk: pd.DataFrame, hrs: pd.DataFrame) -> float:
    vals = []
    if "max_hr_bpm" in wk:
        v = pd.to_numeric(wk["max_hr_bpm"], errors="coerce").dropna()
        if not v.empty:
            vals.append(float(v.max()))
    if hrs is not None and not hrs.empty:
        v = pd.to_numeric(hrs["bpm"], errors="coerce").dropna()
        if not v.empty:
            vals.append(float(np.nanpercentile(v, 99.5)))
    return float(max(vals)) if vals else 0.0


def estimate_hrrest(daily: pd.DataFrame | None, hrs: pd.DataFrame) -> float:
    if daily is not None and not daily.empty and "rhr_bpm" in daily:
        v = pd.to_numeric(daily["rhr_bpm"], errors="coerce").dropna()
        if not v.empty:
            return float(v.median())
    v = pd.to_numeric(hrs["bpm"], errors="coerce").dropna()
    if not v.empty:
        return float(np.nanpercentile(v, 5))
    return 60.0


def zones_trimp_for_workouts(
    hrs: pd.DataFrame, wk: pd.DataFrame, hrmax: float, hrrest: float, sex: str = "M"
) -> pd.DataFrame:
    if hrs is None or hrs.empty or wk is None or wk.empty:
        return pd.DataFrame()
    w = add_effective_fields(wk.copy())
    w["start_utc"] = pd.to_datetime(w.get("start_utc", w.get("start_local")), utc=True)
    w["end_utc"] = pd.to_datetime(w.get("end_utc", w.get("start_local")), utc=True)

    hz = hrs.copy()
    hz["timestamp_utc"] = pd.to_datetime(hz["timestamp_utc"], utc=True)
    hz = hz.sort_values(["workout_id", "timestamp_utc"]).reset_index(drop=True)
    rows: list[dict] = []
    for _, r in w.iterrows():
        wid = r.get("workout_id")
        seg = hz[hz["workout_id"] == wid]
        if seg.empty:
            continue
        # Compute dt_s
        dt = seg["timestamp_utc"].diff().dt.total_seconds().clip(lower=0.5, upper=10).fillna(1.0)
        bpm = pd.to_numeric(seg["bpm"], errors="coerce").ffill().bfill()
        # Zones by percent max
        z1 = dt[(bpm >= 0) & (bpm < 0.6 * hrmax)].sum() / 60.0
        z2 = dt[(bpm >= 0.6 * hrmax) & (bpm < 0.7 * hrmax)].sum() / 60.0
        z3 = dt[(bpm >= 0.7 * hrmax) & (bpm < 0.8 * hrmax)].sum() / 60.0
        z4 = dt[(bpm >= 0.8 * hrmax) & (bpm < 0.9 * hrmax)].sum() / 60.0
        z5 = dt[(bpm >= 0.9 * hrmax)].sum() / 60.0
        trimp = trimp_banister(bpm, hrrest, hrmax, sex=sex)
        rows.append(
            {
                "workout_id": wid,
                "z1_min": float(z1),
                "z2_min": float(z2),
                "z3_min": float(z3),
                "z4_min": float(z4),
                "z5_min": float(z5),
                "trimp": float(trimp),
            }
        )
    return pd.DataFrame(rows)
