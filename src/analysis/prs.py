from __future__ import annotations

import pandas as pd

from .common import add_effective_fields


def pr_by_distance_bucket(
    workouts: pd.DataFrame,
    buckets: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if buckets is None:
        buckets = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 13)]
    w = add_effective_fields(workouts.copy())
    w["start_utc"] = pd.to_datetime(w.get("start_utc", w.get("start_local")), utc=True)
    rows: list[dict] = []
    for lo, hi in buckets:
        subset = w[(w["miles"] >= lo * 0.98) & (w["miles"] <= hi * 1.02)].dropna(
            subset=["pace_min_per_mile_eff"]
        )
        if subset.empty:
            continue
        best = subset.sort_values("pace_min_per_mile_eff").iloc[0]
        rows.append(
            {
                "bucket": f"{lo}-{hi}",
                "workout_id": best["workout_id"] if "workout_id" in best else None,
                "date": best["start_utc"].date().isoformat(),
                "pace_min_per_mile": float(best["pace_min_per_mile_eff"]),
                "miles": float(best["miles"]),
                "activity": best.get("activity"),
            }
        )
    return pd.DataFrame(rows)


def fastest_segments_on_route(
    workouts: pd.DataFrame, routes: pd.DataFrame, precision: float = 0.003
) -> pd.DataFrame:
    if workouts.empty or routes.empty:
        return pd.DataFrame()
    r = routes.copy()
    r["start_lat_c"] = (r["start_lat"] / precision).round().astype(int)
    r["start_lon_c"] = (r["start_lon"] / precision).round().astype(int)
    r["cluster"] = r["start_lat_c"].astype(str) + ":" + r["start_lon_c"].astype(str)
    w = add_effective_fields(workouts.copy())
    j = w.merge(r[["route_id", "cluster"]], on="route_id", how="left").dropna(
        subset=["cluster", "pace_min_per_mile_eff"]
    )
    g = j.groupby("cluster")["pace_min_per_mile_eff"].min().rename("best_pace").reset_index()
    return g.sort_values("best_pace")


def pr_common_races(workouts: pd.DataFrame) -> pd.DataFrame:
    races = [
        ("1mi", 1.0),
        ("5K", 3.1069),
        ("10K", 6.2137),
        ("10mi", 10.0),
        ("Half", 13.1094),
        ("Marathon", 26.219),
    ]
    w = add_effective_fields(workouts.copy())
    w["start_utc"] = pd.to_datetime(w.get("start_utc", w.get("start_local")), utc=True)
    rows: list[dict] = []
    for name, miles in races:
        subset = w[(w["miles"] >= 0.95 * miles) & (w["miles"] <= 1.05 * miles)].dropna(
            subset=["pace_min_per_mile_eff"]
        )
        if subset.empty:
            continue
        best = subset.sort_values("pace_min_per_mile_eff").iloc[0]
        rows.append(
            {
                "race": name,
                "target_miles": miles,
                "date": best["start_utc"].date().isoformat(),
                "pace_min_per_mile": float(best["pace_min_per_mile_eff"]),
                "miles": float(best["miles"]),
                "workout_id": best.get("workout_id"),
            }
        )
    return pd.DataFrame(rows)
