from __future__ import annotations

import pandas as pd


def _pace_min_per_mile(duration_sec: float, distance_m: float | None) -> float | None:
    if not distance_m or distance_m <= 0:
        return None
    miles = float(distance_m) / 1609.344
    if miles <= 0:
        return None
    return (float(duration_sec) / 60.0) / miles


def derive_workout_metrics(
    workouts: pd.DataFrame,
    routes: pd.DataFrame | None = None,
    hr_samples: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = workouts.copy()
    # Prefer GPX moving_time and distance when available
    if routes is not None and not routes.empty and "route_id" in df.columns:
        r = routes.set_index("route_id")
        df["gpx_distance_m"] = (
            df["route_id"].map(r["distance_m"]) if "distance_m" in r.columns else None
        )
        df["gpx_moving_time_sec"] = (
            df["route_id"].map(r["moving_time_sec"]) if "moving_time_sec" in r.columns else None
        )
    # Ensure numerics before fillna to avoid FutureWarning downcasting behavior
    if "gpx_distance_m" in df:
        df["gpx_distance_m"] = pd.to_numeric(df["gpx_distance_m"], errors="coerce")
    if "gpx_moving_time_sec" in df:
        df["gpx_moving_time_sec"] = pd.to_numeric(df["gpx_moving_time_sec"], errors="coerce")
    use_dist = (
        df["gpx_distance_m"].where(df["gpx_distance_m"].notna(), df.get("distance_m"))
        if "gpx_distance_m" in df
        else df.get("distance_m")
    )
    use_time = (
        df["gpx_moving_time_sec"].fillna(df.get("duration_sec"))
        if "gpx_moving_time_sec" in df
        else df.get("duration_sec")
    )

    df["pace_min_per_mile"] = [
        _pace_min_per_mile(t, d) if pd.notna(t) and pd.notna(d) else None
        for t, d in zip(use_time, use_dist, strict=False)
    ]
    df["speed_mph"] = [None if p is None or p == 0 else 60.0 / p for p in df["pace_min_per_mile"]]
    df["beats_per_mile"] = [
        (avg * p) if (pd.notna(avg) and p is not None) else None
        for avg, p in zip(df.get("avg_hr_bpm"), df["pace_min_per_mile"], strict=False)  # type: ignore[arg-type]
    ]
    return df
