from __future__ import annotations

import numpy as np
import pandas as pd

from .common import add_effective_fields


def pace_zscores_by_bucket(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(workouts.copy())
    d = d.dropna(subset=["pace_min_per_mile_eff", "miles"]).copy()
    bins = [0, 2, 3, 4, 5, 6, 8, 13, 20, 100]
    labels = ["<2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-13", "13-20", "20+"]
    d["bucket"] = pd.cut(d["miles"], bins=bins, labels=labels, right=False)
    out = []
    for _b, grp in d.groupby("bucket"):
        if grp.empty:
            continue
        mu = grp["pace_min_per_mile_eff"].mean()
        sd = grp["pace_min_per_mile_eff"].std(ddof=0) or 1.0
        z = (grp["pace_min_per_mile_eff"] - mu) / sd
        tmp = grp[["workout_id", "start_utc", "miles", "pace_min_per_mile_eff"]].copy()
        tmp["z_pace"] = z.values
        out.append(tmp)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not res.empty:
        res["is_exceptional"] = res["z_pace"] <= -2.5
        res["is_struggle"] = res["z_pace"] >= 2.5
    return res


def efficiency_residuals(workouts: pd.DataFrame) -> pd.DataFrame:
    d = add_effective_fields(workouts.copy())
    d["avg_hr_bpm"] = pd.to_numeric(d["avg_hr_bpm"], errors="coerce")
    d["miles"] = pd.to_numeric(d["miles"], errors="coerce")
    d["pace_min_per_mile_eff"] = pd.to_numeric(d["pace_min_per_mile_eff"], errors="coerce")
    d = d.dropna(subset=["pace_min_per_mile_eff", "avg_hr_bpm", "miles"]).copy()
    # Simple linear model: pace ~ avg_hr_bpm + miles
    x = np.c_[np.ones(len(d)), d["avg_hr_bpm"].to_numpy(), d["miles"].to_numpy()]
    y = d["pace_min_per_mile_eff"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    resid = y - y_hat
    out = d[["workout_id", "start_utc", "miles", "avg_hr_bpm", "pace_min_per_mile_eff"]].copy()
    out["residual"] = resid
    q_hi = np.nanpercentile(resid, 97.5)
    q_lo = np.nanpercentile(resid, 2.5)
    out["is_exceptional"] = out["residual"] <= q_lo
    out["is_struggle"] = out["residual"] >= q_hi
    return out


def multivariate_outliers(
    workouts: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    if features is None:
        features = ["pace_min_per_mile_eff", "miles", "avg_hr_bpm"]
    d = add_effective_fields(workouts.copy())
    d = d.dropna(subset=features)
    if d.empty:
        return pd.DataFrame()
    z = (d[features] - d[features].mean()) / d[features].std(ddof=0)
    cov = np.cov(z.T, ddof=0)
    inv = np.linalg.pinv(cov)
    dists = np.einsum("ij,jk,ik->i", z.to_numpy(), inv, z.to_numpy())
    out = d[["workout_id", "start_utc"] + features].copy()
    out["mahal"] = dists
    thr = np.nanpercentile(dists, 99)
    out["is_outlier"] = out["mahal"] >= thr
    return out


def hr_drift(hrs: pd.DataFrame, wk: pd.DataFrame) -> pd.DataFrame:
    if hrs is None or hrs.empty or wk is None or wk.empty:
        return pd.DataFrame()
    h = hrs.copy()
    h["timestamp_utc"] = pd.to_datetime(h["timestamp_utc"], utc=True)
    w = wk.copy()
    w["start_utc"] = pd.to_datetime(w.get("start_utc", w.get("start_local")), utc=True)
    w["end_utc"] = pd.to_datetime(w.get("end_utc", w.get("start_local")), utc=True)
    rows = []
    for _, r in w.iterrows():
        wid = r.get("workout_id")
        seg = h[h["workout_id"] == wid]
        if seg.empty:
            continue
        ts = seg.sort_values("timestamp_utc")
        n = len(ts)
        if n < 20:
            continue
        mid = n // 2
        hr1 = pd.to_numeric(ts["bpm"].iloc[:mid], errors="coerce").mean()
        hr2 = pd.to_numeric(ts["bpm"].iloc[mid:], errors="coerce").mean()
        if pd.isna(hr1) or pd.isna(hr2) or hr1 <= 0:
            continue
        drift = 100.0 * (hr2 - hr1) / hr1
        rows.append({"workout_id": wid, "hr_drift_pct": float(drift)})
    return pd.DataFrame(rows)
