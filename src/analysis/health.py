from __future__ import annotations

import pandas as pd


def daily_rollups(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    out = d.set_index("date").sort_index()
    # Treat zeros as missing for vitals that cannot be 0 physiologically
    for col in [
        "rhr_bpm",
        "hrv_sdnn_ms",
        "vo2max_mlkgmin",
        "walking_hr_avg_bpm",
        "respiratory_rate_bpm",
        "spo2_pct",
    ]:
        if col in out:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where(s > 0)
    if "rhr_bpm" in out:
        out["rhr_7d"] = out["rhr_bpm"].rolling(7, min_periods=2).mean()
    if "hrv_sdnn_ms" in out:
        out["hrv_7d"] = out["hrv_sdnn_ms"].rolling(7, min_periods=2).mean()
    if "vo2max_mlkgmin" in out:
        out["vo2_30d"] = out["vo2max_mlkgmin"].rolling(30, min_periods=5).mean()
    return out.reset_index()


def ecg_counts(ecg: pd.DataFrame) -> pd.DataFrame:
    if ecg is None or ecg.empty or "classification" not in ecg:
        return pd.DataFrame()
    cnt = (
        ecg["classification"]
        .value_counts()
        .rename_axis("classification")
        .rename("count")
        .reset_index()
    )
    return cnt
