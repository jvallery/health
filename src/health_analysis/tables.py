from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write(df: pd.DataFrame, out_dir: Path, name: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, index=False)
    return df


# ---------- Body ----------


def body_weight_daily(body_df: pd.DataFrame, out: Path | None = None, *, to_lb: bool = True) -> pd.DataFrame:
    if body_df is None or body_df.empty or "weight" not in body_df:
        return pd.DataFrame()
    d = body_df[["date", "weight"]].dropna().copy()
    if to_lb:
        d["weight_lb"] = d["weight"].astype(float) / 0.45359237
        d = d.rename(columns={"weight": "weight_kg"})
    name = "body_weight_daily.csv"
    return _write(d, out, name) if out else d


def body_weight_monthly(body_df: pd.DataFrame, out: Path | None = None, *, to_lb: bool = True) -> pd.DataFrame:
    if body_df is None or body_df.empty or "weight" not in body_df:
        return pd.DataFrame()
    d = body_df.copy()
    d["month"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    d["month"] = pd.to_datetime(d["month"]).dt.to_period("M").astype(str)
    if to_lb:
        w = d["weight"].astype(float) / 0.45359237
    else:
        w = d["weight"].astype(float)
    g = (
        d.assign(w=w)
        .groupby("month")["w"]
        .agg(mean="mean", min="min", max="max", count="size")
        .reset_index()
        .rename(columns={"w": "weight"})
    )
    name = "body_weight_monthly.csv"
    return _write(g, out, name) if out else g


def body_fat_daily(body_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if body_df is None or body_df.empty or "body_fat_frac" not in body_df:
        return pd.DataFrame()
    d = (
        body_df[["date", "body_fat_frac"]]
        .dropna()
        .assign(body_fat_pct=lambda x: x["body_fat_frac"].astype(float) * 100.0)
        .drop(columns=["body_fat_frac"])
    )
    name = "body_fat_daily.csv"
    return _write(d, out, name) if out else d


def bmi_daily(body_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if body_df is None or body_df.empty or "bmi" not in body_df:
        return pd.DataFrame()
    d = body_df[["date", "bmi"]].dropna().copy()
    name = "bmi_daily.csv"
    return _write(d, out, name) if out else d


# ---------- Vitals ----------


def resting_hr_daily(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "rhr_bpm" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df[["date", "rhr_bpm"]].dropna().copy()
    name = "resting_hr_daily.csv"
    return _write(d, out, name) if out else d


def resting_hr_monthly(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "rhr_bpm" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df.copy()
    d["month"] = pd.to_datetime(d["date"]).astype("datetime64[ns]").dt.to_period("M").astype(str)
    g = d.groupby("month")["rhr_bpm"].agg(mean="mean", min="min", max="max", count="size").reset_index()
    name = "rhr_monthly.csv"
    return _write(g, out, name) if out else g


def hrv_daily(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "hrv_sdnn_ms" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df[["date", "hrv_sdnn_ms"]].dropna().copy()
    d["hrv_7d"] = d["hrv_sdnn_ms"].rolling(7, min_periods=2).mean()
    name = "hrv_daily.csv"
    return _write(d, out, name) if out else d


def vo2max_daily(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "vo2max_mlkgmin" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df[["date", "vo2max_mlkgmin"]].dropna().copy()
    d["vo2_30d"] = d["vo2max_mlkgmin"].rolling(30, min_periods=5).mean()
    name = "vo2max_daily.csv"
    return _write(d, out, name) if out else d


def bp_daily(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["bp_systolic_mmhg", "bp_diastolic_mmhg"] if c in vitals_df]
    if not cols:
        return pd.DataFrame()
    d = vitals_df[["date"] + cols].dropna(how="all", subset=cols).copy()
    name = "bp_daily.csv"
    return _write(d, out, name) if out else d


def spo2_daily(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "spo2_frac" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df[["date", "spo2_frac"]].dropna().copy()
    d["spo2_pct"] = d["spo2_frac"].astype(float) * 100.0
    d = d.drop(columns=["spo2_frac"])
    name = "spo2_daily.csv"
    return _write(d, out, name) if out else d


def respiratory_daily(vitals_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "resp_rate_bpm" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df[["date", "resp_rate_bpm"]].dropna().copy()
    name = "respiratory_daily.csv"
    return _write(d, out, name) if out else d


# ---------- Energy ----------


def energy_daily(eng_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if eng_df is None or eng_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["active_kcal", "basal_kcal", "steps", "exercise_min"] if c in eng_df]
    if not cols:
        return pd.DataFrame()
    d = eng_df[["date"] + cols].copy()
    name = "active_vs_basal_daily.csv"
    return _write(d, out, name) if out else d


def energy_monthly(eng_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if eng_df is None or eng_df.empty:
        return pd.DataFrame()
    d = eng_df.copy()
    d["month"] = pd.to_datetime(d["date"]).astype("datetime64[ns]").dt.to_period("M").astype(str)
    g = (
        d.groupby("month")[[c for c in ["active_kcal", "basal_kcal"] if c in d]]
        .sum()
        .reset_index()
    )
    name = "active_vs_basal_monthly.csv"
    return _write(g, out, name) if out else g


# ---------- Sleep ----------


def sleep_daily(sleep_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if sleep_df is None or sleep_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["sleep_in_bed_min", "sleep_asleep_min", "sleep_awake_min", "sleep_efficiency"] if c in sleep_df]
    if not cols:
        return pd.DataFrame()
    d = sleep_df[["date"] + cols].copy()
    name = "sleep_daily.csv"
    return _write(d, out, name) if out else d


def sleep_efficiency_daily(sleep_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if sleep_df is None or sleep_df.empty or "sleep_efficiency" not in sleep_df:
        return pd.DataFrame()
    d = sleep_df[["date", "sleep_efficiency"]].copy()
    name = "sleep_efficiency.csv"
    return _write(d, out, name) if out else d


# ---------- Alerts & Audio ----------


def hr_alerts_monthly(alerts_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame()
    d = alerts_df.copy()
    name = "hr_alerts.csv"
    return _write(d, out, name) if out else d


def audio_exposure_weekly(audio_df: pd.DataFrame, out: Path | None = None) -> pd.DataFrame:
    if audio_df is None or audio_df.empty:
        return pd.DataFrame()
    d = audio_df.copy()
    name = "audio_exposure.csv"
    return _write(d, out, name) if out else d


# ---------- Derived ----------


def weight_slope(body_df: pd.DataFrame, *, to_lb: bool = True) -> float | None:
    if body_df is None or body_df.empty or "weight" not in body_df:
        return None
    d = body_df.dropna(subset=["weight"]).copy()
    if d.empty:
        return None
    y = d["weight"].astype(float).to_numpy()
    if to_lb:
        y = y / 0.45359237
    x = pd.to_datetime(d["date"]).map(pd.Timestamp.toordinal).to_numpy().astype(float)
    if len(x) < 7:
        return None
    slope_per_day = float(np.polyfit(x, y, 1)[0])
    return slope_per_day * 7.0  # per week


def rhr_vo2_corr(vitals_df: pd.DataFrame) -> tuple[float | None, int]:
    if vitals_df is None or vitals_df.empty:
        return None, 0
    cols = ["rhr_bpm", "vo2max_mlkgmin"]
    if any(c not in vitals_df for c in cols):
        return None, 0
    a = pd.to_numeric(vitals_df["rhr_bpm"], errors="coerce")
    b = pd.to_numeric(vitals_df["vo2max_mlkgmin"], errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return None, int(mask.sum())
    r = float(a[mask].corr(b[mask], method="pearson"))
    return r, int(mask.sum())


def hrv_low_flags(vitals_df: pd.DataFrame, threshold_ms: float = 20.0, out: Path | None = None) -> pd.DataFrame:
    if vitals_df is None or vitals_df.empty or "hrv_sdnn_ms" not in vitals_df:
        return pd.DataFrame()
    d = vitals_df[["date", "hrv_sdnn_ms"]].dropna().copy()
    d["hrv_low_flag"] = d["hrv_sdnn_ms"].astype(float) < float(threshold_ms)
    name = "hrv_low_flags.csv"
    return _write(d, out, name) if out else d

