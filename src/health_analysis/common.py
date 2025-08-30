from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from healthkit.parsing.export_xml import parse_records as _parse_records
from healthkit.utils.time import to_local
from healthkit.utils.units import to_kcal


VITAL_MEAN = {
    "RestingHeartRate": "rhr_bpm",
    "WalkingHeartRateAverage": "walking_hr_avg_bpm",
    "HeartRateVariabilitySDNN": "hrv_sdnn_ms",
    "RespiratoryRate": "resp_rate_bpm",
    "OxygenSaturation": "spo2_frac",
}

VITAL_MAX = {
    "VO2Max": "vo2max_mlkgmin",
}

SUM_TYPES = {
    "StepCount": "steps",
    "ActiveEnergyBurned": "active_kcal",
    "BasalEnergyBurned": "basal_kcal",
    "AppleExerciseTime": "exercise_min",
}

BODY_TYPES = {
    "BodyMass": "weight",  # normalize to kg internally
    "BodyFatPercentage": "body_fat_frac",  # 0..1
    "BodyMassIndex": "bmi",
}

BP_TYPES = {
    "BloodPressureSystolic": "bp_systolic_mmhg",
    "BloodPressureDiastolic": "bp_diastolic_mmhg",
}

SLEEP_TYPE = "SleepAnalysis"
ALERT_TYPES = {"HighHeartRateEvent": "high", "LowHeartRateEvent": "low"}
AUDIO_TYPES = {
    "HeadphoneAudioExposure": "headphone",
    "EnvironmentalAudioExposure": "environment",
}


@dataclass
class HealthConfig:
    tz: str = "America/Denver"
    weight_goal_lb: float | None = None
    # output units
    out_weight_lb: bool = True


def parse_records(xml_path: str, type_whitelist: list[str]) -> pd.DataFrame:
    """
    Wrap healthkit.parse_records and return a single tidy dataframe with columns:
    type, start_utc, end_utc, start_local, end_local, date, value (float), unit, source_name.
    """
    buckets = _parse_records(xml_path, types=type_whitelist)
    frames: list[pd.DataFrame] = []
    for rtype, df in buckets.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["type"] = rtype
        d["start_local"] = to_local(d["start_utc"], HealthConfig.tz) if hasattr(HealthConfig, "tz") else d[
            "start_utc"
        ]
        d["end_local"] = to_local(d["end_utc"], HealthConfig.tz) if hasattr(HealthConfig, "tz") else d[
            "end_utc"
        ]
        d["date"] = pd.to_datetime(d["start_utc"], utc=True).dt.tz_convert(HealthConfig.tz).dt.date
        # Coerce value to numeric when possible
        d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=[
            "type",
            "start_utc",
            "end_utc",
            "start_local",
            "end_local",
            "date",
            "value",
            "value_num",
            "unit",
            "source_name",
        ])
    out = pd.concat(frames, ignore_index=True)
    return out


def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Weight normalization to kg
    mask_w = d["type"] == "BodyMass"
    if mask_w.any():
        u = d.loc[mask_w, "unit"].astype(str).str.lower()
        vals = d.loc[mask_w, "value_num"].astype(float)
        kg = np.where(u.str.contains("lb"), vals * 0.45359237, vals)
        d.loc[mask_w, "value_num"] = kg
        d.loc[mask_w, "unit"] = "kg"
    # Energies to kcal
    for t in ["ActiveEnergyBurned", "BasalEnergyBurned"]:
        m = d["type"] == t
        if m.any():
            d.loc[m, "value_num"] = [to_kcal(v, u) or 0.0 for v, u in zip(d.loc[m, "value"], d.loc[m, "unit"])]
            d.loc[m, "unit"] = "kcal"
    # Body fat percentage ensure 0..1
    m = d["type"] == "BodyFatPercentage"
    if m.any():
        # Apple exports as 0.xx with unit "%" — keep numeric fraction
        d.loc[m, "value_num"] = pd.to_numeric(d.loc[m, "value_num"], errors="coerce").clip(lower=0, upper=1)
        d.loc[m, "unit"] = "fraction"
    # Oxygen saturation 0..1
    m = d["type"] == "OxygenSaturation"
    if m.any():
        d.loc[m, "value_num"] = pd.to_numeric(d.loc[m, "value_num"], errors="coerce").clip(lower=0, upper=1)
        d.loc[m, "unit"] = "fraction"
    return d


def daily_aggregates(records_df: pd.DataFrame, tz: str) -> dict[str, pd.DataFrame]:
    """
    Build per-domain daily tables.
    Returns a dict of category -> dataframe with at minimum a 'date' column.
    """
    if records_df is None or records_df.empty:
        return {k: pd.DataFrame() for k in ["body", "vitals", "energy", "sleep", "alerts", "audio"]}

    d = records_df.copy()
    # Fix tz for date derivation
    d["date"] = pd.to_datetime(d["start_utc"], utc=True).dt.tz_convert(tz).dt.date
    d = _normalize_units(d)

    out: dict[str, pd.DataFrame] = {}

    # Body
    body = []
    for t, col in BODY_TYPES.items():
        sub = d[d["type"] == t]
        if sub.empty:
            continue
        g = sub.groupby("date")["value_num"].mean().rename(col)
        body.append(g)
    out["body"] = pd.concat(body, axis=1).reset_index() if body else pd.DataFrame(columns=["date"])

    # Vitals (mean), BP (mean), VO2 (max)
    vit_parts = []
    for t, col in VITAL_MEAN.items():
        sub = d[d["type"] == t]
        if sub.empty:
            continue
        g = sub.groupby("date")["value_num"].mean().rename(col)
        vit_parts.append(g)
    for t, col in VITAL_MAX.items():
        sub = d[d["type"] == t]
        if sub.empty:
            continue
        g = sub.groupby("date")["value_num"].max().rename(col)
        vit_parts.append(g)
    # Blood pressure
    if (d["type"] == "BloodPressureSystolic").any():
        vit_parts.append(
            d[d["type"] == "BloodPressureSystolic"].groupby("date")["value_num"].mean().rename("bp_systolic_mmhg")
        )
    if (d["type"] == "BloodPressureDiastolic").any():
        vit_parts.append(
            d[d["type"] == "BloodPressureDiastolic"].groupby("date")["value_num"].mean().rename("bp_diastolic_mmhg")
        )
    out["vitals"] = pd.concat(vit_parts, axis=1).reset_index() if vit_parts else pd.DataFrame(columns=["date"])

    # Energy (sum)
    eng_parts = []
    for t, col in SUM_TYPES.items():
        sub = d[d["type"] == t]
        if sub.empty:
            continue
        g = sub.groupby("date")["value_num"].sum().rename(col)
        eng_parts.append(g)
    out["energy"] = pd.concat(eng_parts, axis=1).reset_index() if eng_parts else pd.DataFrame(columns=["date"])

    # Sleep — sum minutes by category (coarse: attribute to start day)
    sleep = d[d["type"] == SLEEP_TYPE].copy()
    if not sleep.empty:
        # duration minutes
        dur = (
            pd.to_datetime(sleep["end_utc"], utc=True) - pd.to_datetime(sleep["start_utc"], utc=True)
        ).dt.total_seconds() / 60.0
        sleep["minutes"] = dur.clip(lower=0)
        sleep["val"] = sleep["value"].astype(str)
        asleep = sleep[sleep["val"].str.contains("Asleep", case=False, na=False)]
        inbed = sleep[sleep["val"].str.contains("InBed", case=False, na=False)]
        awake = sleep[sleep["val"].str.contains("Awake", case=False, na=False)]
        parts = []
        if not inbed.empty:
            parts.append(inbed.groupby("date")["minutes"].sum().rename("sleep_in_bed_min"))
        if not asleep.empty:
            parts.append(asleep.groupby("date")["minutes"].sum().rename("sleep_asleep_min"))
        if not awake.empty:
            parts.append(awake.groupby("date")["minutes"].sum().rename("sleep_awake_min"))
        sleep_daily = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        if not sleep_daily.empty:
            sleep_daily["sleep_efficiency"] = (
                sleep_daily["sleep_asleep_min"] / sleep_daily["sleep_in_bed_min"]
            ).clip(upper=1.0)
            sleep_daily = sleep_daily.reset_index()
        out["sleep"] = sleep_daily if not sleep_daily.empty else pd.DataFrame(columns=["date"])
    else:
        out["sleep"] = pd.DataFrame(columns=["date"])

    # Alerts — count per month
    alerts = d[d["type"].isin(ALERT_TYPES.keys())].copy()
    if not alerts.empty:
        alerts["month"] = pd.to_datetime(alerts["start_utc"], utc=True).dt.tz_convert(tz).dt.to_period("M").astype(str)
        alerts["kind"] = alerts["type"].map(ALERT_TYPES)
        out["alerts"] = (
            alerts.groupby(["month", "kind"]).size().rename("count").reset_index()
        )
    else:
        out["alerts"] = pd.DataFrame(columns=["month", "kind", "count"])

    # Audio exposure — weekly counts and avg dB
    audio = d[d["type"].isin(AUDIO_TYPES.keys())].copy()
    if not audio.empty:
        audio["week"] = pd.to_datetime(audio["start_utc"], utc=True).dt.tz_convert(tz).dt.to_period("W-MON").dt.start_time
        audio["kind"] = audio["type"].map(AUDIO_TYPES)
        agg = audio.groupby(["week", "kind"]).agg(events=("value", "size"), avg_db=("value_num", "mean"))
        out["audio"] = agg.reset_index()
    else:
        out["audio"] = pd.DataFrame(columns=["week", "kind", "events", "avg_db"])

    return out


def join_daily_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f.set_index("date") for f in frames if f is not None and not f.empty and "date" in f]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.reset_index()

