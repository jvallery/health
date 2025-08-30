from __future__ import annotations

import pandas as pd

from .common import add_effective_fields


def _ensure_times(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "start_dt_local" in d:
        d["start_local"] = pd.to_datetime(d["start_dt_local"], utc=True, errors="coerce")
    elif "start_local" in d:
        d["start_local"] = pd.to_datetime(d["start_local"], utc=True, errors="coerce")
    else:
        d["start_local"] = pd.to_datetime(d["start_utc"], utc=True, errors="coerce")
    return d


def monthly_perf(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(_ensure_times(workouts))
    d["month"] = d["start_local"].dt.strftime("%Y-%m")
    g = d.groupby("month", dropna=True)
    out = pd.DataFrame(
        {
            "avg_pace": g["pace_min_per_mile_eff"].mean(),
            "median_pace": g["pace_min_per_mile_eff"].median(),
            "avg_miles_per_workout": g["miles"].mean(),
            "total_miles": g["miles"].sum(),
            "workouts": g.size(),
        }
    ).reset_index()
    return out


def yearly_perf(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(_ensure_times(workouts))
    d["year"] = d["start_local"].dt.year
    g = d.groupby("year", dropna=True)
    out = pd.DataFrame(
        {
            "avg_pace": g["pace_min_per_mile_eff"].mean(),
            "median_pace": g["pace_min_per_mile_eff"].median(),
            "avg_miles_per_workout": g["miles"].mean(),
            "total_miles": g["miles"].sum(),
            "workouts": g.size(),
        }
    ).reset_index()
    return out


def efficiency_trend(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(_ensure_times(workouts))
    if "beats_per_mile" not in d or d["beats_per_mile"].isna().all():
        # Robustly backfill avg_hr_bpm from samples when available
        if "avg_hr_bpm" not in d and "avg_hr_bpm_from_samples" in d:
            d["avg_hr_bpm"] = d["avg_hr_bpm_from_samples"]
        elif "avg_hr_bpm_from_samples" in d:
            d["avg_hr_bpm"] = d["avg_hr_bpm"].fillna(d["avg_hr_bpm_from_samples"])
        if "avg_hr_bpm" in d:
            d["beats_per_mile"] = d["avg_hr_bpm"] * d["pace_min_per_mile_eff"]
    d["month"] = d["start_local"].dt.strftime("%Y-%m")
    g = (
        d.groupby("month", dropna=True)["beats_per_mile"]
        .median()
        .rename("median_beats_per_mile")
        .reset_index()
    )
    g["median_beats_per_mile_roll3"] = g["median_beats_per_mile"].rolling(3, min_periods=1).median()
    return g


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def pace_by_season(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(_ensure_times(workouts))
    d["season"] = d["start_local"].dt.month.map(_season_from_month)
    g = d.groupby("season", dropna=True).agg(
        median_pace=("pace_min_per_mile_eff", "median"),
        median_miles=("miles", "median"),
        n=("miles", "size"),
    )
    return g.reset_index()


def time_of_day_perf(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(_ensure_times(workouts))
    hour = d["start_local"].dt.hour
    bucket = pd.cut(
        hour,
        bins=[-1, 5, 10, 15, 21, 24],
        labels=["night", "morning", "midday", "evening", "late"],
        right=False,
    )
    d["tod_bucket"] = bucket
    g = d.groupby("tod_bucket", dropna=True).agg(
        median_pace=("pace_min_per_mile_eff", "median"),
        median_hr=("avg_hr_bpm", "median"),
        workouts=("miles", "size"),
        miles=("miles", "sum"),
    )
    return g.reset_index()


def weekly_volume(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(_ensure_times(workouts))
    d["week"] = d["start_local"].dt.to_period("W-MON").dt.start_time
    g = d.groupby("week", dropna=True).agg(miles=("miles", "sum"), workouts=("miles", "size"))
    return g.reset_index()


def calendar_heatmap_data(
    workouts: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, list[int], list[int]]:
    d = add_effective_fields(_ensure_times(workouts))
    dt = d["start_local"].dt.tz_convert(None)
    d = d.assign(date=dt.dt.date, year=dt.dt.year)
    dy = d[d["year"] == year]
    daily = dy.groupby("date")["miles"].sum()
    # Build 7 x 53 matrix (weeks start Monday)
    import datetime as _dt

    start = _dt.date(year, 1, 1)
    # Align to Monday
    start -= _dt.timedelta(days=(start.weekday()))
    weeks = 54
    matrix = [[0.0 for _ in range(weeks)] for __ in range(7)]
    dates = []
    for w in range(weeks):
        for dows in range(7):
            cur = start + _dt.timedelta(days=w * 7 + dows)
            dates.append(cur)
            if cur.year == year and cur in daily.index:
                matrix[dows][w] = float(daily[cur])
    import pandas as _pd

    mat = _pd.DataFrame(matrix, index=list(range(7)), columns=list(range(weeks)))
    return mat, list(range(weeks)), list(range(7))


def streaks_summary(workouts: pd.DataFrame, n_per_week: int = 3) -> dict:
    d = _ensure_times(workouts)
    days = sorted(pd.to_datetime(d["start_local"], utc=True).dt.date.unique())
    # Longest day streak
    longest_days = cur = 0
    prev = None
    for day in days:
        if prev is None or (day - prev).days == 1:
            cur += 1
        else:
            cur = 1
        longest_days = max(longest_days, cur)
        prev = day

    # Weeks with >= n workouts
    w = d.copy()
    w["week"] = pd.to_datetime(w["start_local"], utc=True).dt.to_period("W-MON").dt.start_time
    weekly_counts = w.groupby("week").size()
    mask = (weekly_counts >= n_per_week).astype(int)
    # Longest consecutive weeks meeting threshold
    longest_weeks = 0
    cur = 0
    for val in mask:
        if val:
            cur += 1
        else:
            cur = 0
        longest_weeks = max(longest_weeks, cur)

    return {"longest_day_streak": int(longest_days), "longest_week_streak_ge_n": int(longest_weeks)}
