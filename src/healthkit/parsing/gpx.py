from __future__ import annotations

import hashlib
import os
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd

from ..utils import geo


def _route_id_for(path: str) -> str:
    return hashlib.sha1(os.path.basename(path).encode()).hexdigest()


def parse_gpx_file(path: str) -> tuple[dict, pd.DataFrame]:
    with open(path, encoding="utf-8") as f:
        g = gpxpy.parse(f)
    pts: list[tuple] = []
    for trk in g.tracks:
        for seg in trk.segments:
            for p in seg.points:
                # gpxpy times are aware (UTC) or naive (assume UTC)
                if getattr(p.time, "tzinfo", None) is not None:
                    t = pd.to_datetime(p.time).tz_convert("UTC")
                else:
                    t = pd.to_datetime(p.time, utc=True)
                pts.append(
                    (
                        t,
                        float(p.latitude),
                        float(p.longitude),
                        float(getattr(p, "elevation", 0.0) or 0.0),
                    )
                )
    if not pts:
        return {}, pd.DataFrame()
    df = pd.DataFrame(pts, columns=["time_utc", "lat", "lon", "ele_m"])
    df = df.sort_values("time_utc").reset_index(drop=True)
    df["d_m"] = geo.haversine_series(df["lat"], df["lon"])  # per-point delta
    df["dt_s"] = df["time_utc"].diff().dt.total_seconds().clip(lower=0).fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed_mps"] = (df["d_m"] / df["dt_s"]).replace([np.inf, -np.inf], 0).fillna(0)
    moving = df["speed_mps"] > 0.5
    dist_m = float(df["d_m"].sum())
    moving_time = float(df.loc[moving, "dt_s"].sum())
    gain, loss = geo.elevation_gain_loss(df["ele_m"].to_numpy(), threshold=3.0)
    rid = _route_id_for(path)
    route = {
        "route_id": rid,
        "file": os.path.basename(path),
        "start_utc": df["time_utc"].min(),
        "end_utc": df["time_utc"].max(),
        "start_lat": df["lat"].iloc[0],
        "start_lon": df["lon"].iloc[0],
        "end_lat": df["lat"].iloc[-1],
        "end_lon": df["lon"].iloc[-1],
        "points": int(len(df)),
        "distance_m": dist_m,
        "moving_time_sec": moving_time,
        "elev_gain_m": gain,
        "elev_loss_m": loss,
    }
    return route, df.assign(route_id=rid)


def parse_gpx_dir(dir_path: str | Path, with_points: bool = False):
    p = Path(dir_path)
    if not p.exists():
        return pd.DataFrame(), None
    routes: list[dict] = []
    points: list[pd.DataFrame] = []
    for f in sorted(p.rglob("*.gpx")):
        route, pts = parse_gpx_file(str(f))
        if route:
            routes.append(route)
            if with_points and not pts.empty:
                points.append(pts)
    routes_df = pd.DataFrame(routes)
    points_df = pd.concat(points, ignore_index=True) if points else None
    return routes_df, points_df
