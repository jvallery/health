from __future__ import annotations

import pandas as pd


def annotate_weather(
    workouts: pd.DataFrame, routes: pd.DataFrame, tz: str = "America/Denver"
) -> pd.DataFrame:
    try:
        from meteostat import Hourly, Point  # type: ignore
    except Exception:
        return pd.DataFrame()  # meteostat not installed

    if workouts.empty or routes.empty:
        return pd.DataFrame()
    r = routes[["route_id", "start_lat", "start_lon"]].dropna()
    w = workouts.merge(r, on="route_id", how="left")
    start = pd.to_datetime(w.get("start_local", w.get("start_utc")), utc=True)
    w["start_local"] = start.dt.tz_convert(tz)
    rows = []
    for _, row in w.dropna(subset=["start_lat", "start_lon"]).iterrows():
        loc = Point(float(row["start_lat"]), float(row["start_lon"]))
        t = row["start_local"]
        try:
            df = Hourly(loc, t.floor("H"), t.ceil("H")).fetch()
        except Exception:
            continue
        if df.empty:
            continue
        h = df.iloc[0]
        rows.append(
            {
                "workout_id": row.get("workout_id"),
                "temp_c": float(h.get("temp") or 0.0),
                "dwpt_c": float(h.get("dwpt") or 0.0),
                "rh": float(h.get("rhum") or 0.0),
                "wind_mps": float((h.get("wspd") or 0.0) / 3.6),
                "precip_mm": float(h.get("prcp") or 0.0),
                "pressure_hpa": float(h.get("pres") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def weather_effects(workouts_with_weather: pd.DataFrame) -> pd.DataFrame:
    if workouts_with_weather.empty:
        return pd.DataFrame()
    d = workouts_with_weather.copy()
    d["temp_bucket_f"] = pd.cut(
        d["temp_c"] * 9 / 5 + 32,
        bins=[-100, 32, 40, 50, 60, 70, 80, 90, 200],
        labels=["<32", "32-40", "40-50", "50-60", "60-70", "70-80", "80-90", ">90"],
        right=False,
    )
    baseline = d[d["temp_bucket_f"].isin(["50-60"])]["pace_min_per_mile_eff"].median()
    g = (
        d.groupby("temp_bucket_f")
        .agg(median_pace=("pace_min_per_mile_eff", "median"), n=("pace_min_per_mile_eff", "size"))
        .reset_index()
    )
    g["delta_vs_50_60F"] = g["median_pace"] - baseline
    return g
