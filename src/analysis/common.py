from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def add_effective_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Prefer GPX-derived fields when present
    dist = (
        d.get("gpx_distance_m").where(d.get("gpx_distance_m").notna(), d.get("distance_m"))
        if "gpx_distance_m" in d
        else d.get("distance_m")
    )
    move = (
        d.get("gpx_moving_time_sec").where(
            d.get("gpx_moving_time_sec").notna(), d.get("duration_sec")
        )
        if "gpx_moving_time_sec" in d
        else d.get("duration_sec")
    )
    d = d.assign(
        distance_m_eff=pd.to_numeric(dist, errors="coerce"),
        moving_time_sec_eff=pd.to_numeric(move, errors="coerce"),
    )
    d["miles"] = d["distance_m_eff"] / 1609.34
    if "pace_min_per_mile" in d.columns and d["pace_min_per_mile"].notna().any():
        d["pace_min_per_mile_eff"] = pd.to_numeric(d["pace_min_per_mile"], errors="coerce")
    else:
        denom = d["miles"].replace(0, np.nan)
        d["pace_min_per_mile_eff"] = (d["moving_time_sec_eff"] / 60.0) / denom
    return d


def normalize_workouts(df: pd.DataFrame, user_tz: str) -> pd.DataFrame:
    tz = ZoneInfo(user_tz)
    d = add_effective_fields(df)

    # Time normalization
    start_local = df.get("start_local")
    start_utc = (
        pd.to_datetime(df.get("start_utc"), utc=True, errors="coerce")
        if "start_utc" in df
        else None
    )
    if start_local is not None:
        s = pd.to_datetime(start_local, errors="coerce")
        if getattr(s.dt, "tz", None) is not None:
            start_dt_local = s.dt.tz_convert(tz)
        else:
            # localize naive timestamps to user tz
            start_dt_local = s.dt.tz_localize(tz)
    elif start_utc is not None is not False:
        start_dt_local = start_utc.dt.tz_convert(tz)
    else:
        start_dt_local = pd.to_datetime("1970-01-01", utc=True).tz_convert(tz)  # fallback scalar

    d["start_dt_local"] = start_dt_local
    d["date_local"] = d["start_dt_local"].dt.date
    d["year"] = d["start_dt_local"].dt.year
    d["month"] = d["start_dt_local"].dt.month
    d["hour"] = d["start_dt_local"].dt.hour
    d["dow"] = d["start_dt_local"].dt.day_name()
    # Week start (Monday) in local tz; keep tz-aware
    d["week_start"] = d["start_dt_local"].dt.to_period("W-MON").dt.start_time

    # Filter degenerate
    d = d[(d["miles"] > 0.05) & d["moving_time_sec_eff"].notna()].copy()
    return d


def mmss(pace_min: float | None) -> str:
    if pace_min is None or pd.isna(pace_min):
        return ""
    m = int(pace_min)
    s = int(round((pace_min - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:d}:{s:02d}"
