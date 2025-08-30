from __future__ import annotations

import pandas as pd

from .common import add_effective_fields


def route_start_clusters(routes: pd.DataFrame, precision: float = 0.003) -> pd.DataFrame:
    if routes.empty:
        return pd.DataFrame(columns=["cluster", "n"])
    r = routes.copy()
    r["start_lat_c"] = (r["start_lat"] / precision).round().astype(int)
    r["start_lon_c"] = (r["start_lon"] / precision).round().astype(int)
    r["cluster"] = r["start_lat_c"].astype(str) + ":" + r["start_lon_c"].astype(str)
    g = r.groupby("cluster").size().rename("n").reset_index().sort_values("n", ascending=False)
    return g


def join_elevation(wk: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    if wk.empty or routes.empty:
        return pd.DataFrame()
    w = add_effective_fields(wk.copy())
    j = w.merge(routes[["route_id", "elev_gain_m"]], on="route_id", how="left")
    j["elev_per_mile"] = j["elev_gain_m"] / j["miles"].replace(0, pd.NA)
    return j


def cluster_yearly_trends(
    wk: pd.DataFrame, routes: pd.DataFrame, top_n: int = 3
) -> list[pd.DataFrame]:
    perf = route_cluster_perf(wk, routes)
    if perf.empty:
        return []
    tops = (
        perf.groupby("cluster")["n"].sum()
        if "n" in perf.columns
        else perf.groupby("cluster")["median_pace"].count()
    )
    clusters = list(tops.sort_values(ascending=False).head(top_n).index)
    out = []
    for c in clusters:
        out.append(perf[perf["cluster"] == c].copy())
    return out


def route_cluster_perf(
    workouts: pd.DataFrame, routes: pd.DataFrame, precision: float = 0.003
) -> pd.DataFrame:
    if workouts.empty or routes.empty:
        return pd.DataFrame()
    r = routes.copy()
    r["start_lat_c"] = (r["start_lat"] / precision).round().astype(int)
    r["start_lon_c"] = (r["start_lon"] / precision).round().astype(int)
    r["cluster"] = r["start_lat_c"].astype(str) + ":" + r["start_lon_c"].astype(str)
    w = add_effective_fields(workouts.copy())
    j = w.merge(r[["route_id", "cluster"]], on="route_id", how="left")
    j["year"] = pd.to_datetime(j.get("start_local", j.get("start_utc")), utc=True).dt.year
    g = (
        j.groupby(["cluster", "year"])
        .agg(
            median_pace=("pace_min_per_mile_eff", "median"),
            median_miles=("miles", "median"),
            n=("miles", "size"),
        )
        .reset_index()
        .sort_values(["cluster", "year"])
    )
    return g


def annotate_cluster_labels(
    clusters: pd.DataFrame,
    precision: float,
    cache_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Annotate clusters with human-readable place labels using reverse geocoding.

    - Uses OpenStreetMap Nominatim (public endpoint) with very light usage (top clusters only).
    - Respects an optional cache DataFrame with columns [cluster, label].
    - If geocoding fails, falls back to formatted lat,lon at given precision.
    """
    if clusters is None or clusters.empty:
        return clusters
    import json
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    cached: dict[str, str] = {}
    if cache_df is not None and not cache_df.empty:
        for _, r in cache_df.iterrows():
            cached[str(r["cluster"])] = str(r["label"]) if pd.notna(r["label"]) else ""

    def clus_to_latlon(c: str) -> tuple[float, float]:
        a, b = str(c).split(":")
        return (int(a) * precision, int(b) * precision)

    labels: list[str] = []
    import re as _re
    for c in clusters["cluster"].astype(str):
        cached_label = cached.get(c, "")
        def _is_generic(lbl: str) -> bool:
            # e.g., "Longmont, Colorado" or single token lat,lon
            if not lbl:
                return True
            if _re.match(r"^[^,]+,\s*[A-Za-z .]+$", lbl):
                return True
            return False
        # Re‑geocode if missing or too generic
        if cached_label and not _is_generic(cached_label):
            labels.append(cached_label)
            continue
        lat, lon = clus_to_latlon(c)
        # Build request
        params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 17, "addressdetails": 1}
        url = "https://nominatim.openstreetmap.org/reverse?" + urlencode(params)
        ua = "apple-health-analytics/0.1 (github; training analytics)"
        try:
            req = Request(url, headers={"User-Agent": ua})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            addr = (data or {}).get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village")
            hood = addr.get("neighbourhood") or addr.get("suburb") or addr.get("hamlet")
            road = addr.get("road") or addr.get("residential")
            name = (data or {}).get("name")
            state = addr.get("state")
            label = None
            # Prefer POI or road + neighbourhood for specificity
            if name and city:
                label = f"{name}, {city}"
            elif road and hood and city:
                label = f"{road}, {hood} ({city})"
            elif hood and city:
                label = f"{hood}, {city}"
            elif city and state:
                label = f"{city}, {state}"
            elif city:
                label = city
            else:
                label = f"{lat:.3f}, {lon:.3f}"
        except Exception:
            label = f"{lat:.3f}, {lon:.3f}"
        labels.append(label)
        cached[c] = label

    out = clusters.copy()
    out["label"] = labels
    return out
