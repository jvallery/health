from __future__ import annotations

import pandas as pd


def build_qa_flags(workouts: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if workouts is None or workouts.empty:
        return pd.DataFrame()
    w = workouts.copy()
    w["start_utc"] = pd.to_datetime(w["start_utc"], utc=True)
    if routes is None or routes.empty:
        # Every workout without route
        for _, r in w.iterrows():
            if not r.get("has_route"):
                rows.append(
                    {
                        "table": "workouts",
                        "id": r["workout_id"],
                        "flag": "route_missing",
                        "value": None,
                        "details": None,
                    }
                )
        return pd.DataFrame(rows)

    r = routes.copy()
    r["start_utc"] = pd.to_datetime(r["start_utc"], utc=True)
    r_index = r.set_index("route_id")

    # Distance mismatch on linked
    for _, wk in w.iterrows():
        rid = wk.get("route_id")
        if pd.isna(rid):
            rows.append(
                {
                    "table": "workouts",
                    "id": wk["workout_id"],
                    "flag": "route_missing",
                    "value": None,
                    "details": None,
                }
            )
            continue
        if rid not in r_index.index:
            rows.append(
                {
                    "table": "workouts",
                    "id": wk["workout_id"],
                    "flag": "route_id_not_found",
                    "value": str(rid),
                    "details": None,
                }
            )
            continue
        gpx_dist = r_index.at[rid, "distance_m"] if "distance_m" in r_index.columns else None
        hk_dist = wk.get("distance_m")
        if pd.notna(gpx_dist) and pd.notna(hk_dist) and hk_dist:
            pct = abs(float(gpx_dist) - float(hk_dist)) / float(hk_dist)
            if pct > 0.15:
                rows.append(
                    {
                        "table": "workouts",
                        "id": wk["workout_id"],
                        "flag": "distance_mismatch_pct",
                        "value": float(round(pct, 4)),
                        "details": None,
                    }
                )

    # Unmatched routes (no workout referencing them)
    linked = set(w.dropna(subset=["route_id"])["route_id"].astype(str))
    for _, rr in r.iterrows():
        if str(rr["route_id"]) not in linked:
            rows.append(
                {
                    "table": "routes",
                    "id": rr["route_id"],
                    "flag": "unmatched_route",
                    "value": None,
                    "details": rr.get("file"),
                }
            )

    return pd.DataFrame(rows)
