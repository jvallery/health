from __future__ import annotations

import pandas as pd


def link_routes(workouts: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    if workouts.empty or routes.empty:
        return workouts
    w = workouts.copy()
    r = routes.copy()
    w["start_utc"] = pd.to_datetime(w["start_utc"], utc=True)
    r["start_utc"] = pd.to_datetime(r["start_utc"], utc=True)

    # Build time window index for fast lookup
    r = r.sort_values("start_utc").reset_index(drop=True)
    matched_route_id = []
    matched_has_route = []
    for _, row in w.iterrows():
        start = row["start_utc"]
        window = r[
            (r["start_utc"] >= start - pd.Timedelta(minutes=30))
            & (r["start_utc"] <= start + pd.Timedelta(minutes=30))
        ]
        if window.empty:
            matched_route_id.append(None)
            matched_has_route.append(False)
            continue
        # If workout has distance, prefer closest distance diff
        if pd.notna(row.get("distance_m")) and row.get("distance_m"):
            ww = window.copy()
            ww["dist_diff"] = (ww["distance_m"] - float(row["distance_m"])).abs() / float(
                row["distance_m"]
            )
            pick = ww.sort_values(["dist_diff", "start_utc"]).iloc[0]
        else:
            pick = window.sort_values("start_utc").iloc[0]
        matched_route_id.append(pick["route_id"])
        matched_has_route.append(True)
    w["route_id"] = matched_route_id
    w["has_route"] = matched_has_route
    return w
