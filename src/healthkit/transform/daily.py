from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
from lxml import etree

from ..utils.units import to_kcal


def daily_summary(records_by_type: dict[str, pd.DataFrame], tz_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rtype, df in records_by_type.items():
        if df.empty:
            continue
        d = df.copy()
        d["start_local"] = pd.to_datetime(d["start_utc"], utc=True).dt.tz_convert(tz_name)
        d["date"] = d["start_local"].dt.date
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        agg = None
        if rtype in {"StepCount", "ActiveEnergyBurned", "BasalEnergyBurned", "AppleExerciseTime"}:
            agg = d.groupby("date")["value"].sum().rename(rtype)
        elif rtype in {"RestingHeartRate", "WalkingHeartRateAverage", "HeartRateVariabilitySDNN"}:
            agg = d.groupby("date")["value"].mean().rename(rtype)
        elif rtype in {"VO2Max"}:
            agg = d.groupby("date")["value"].max().rename(rtype)
        if agg is not None:
            frames.append(agg.to_frame())
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.reset_index()
    return out


def daily_summary_stream(xml_path: str, tz_name: str) -> pd.DataFrame:
    tz = ZoneInfo(tz_name)
    sum_types = {"StepCount", "ActiveEnergyBurned", "BasalEnergyBurned", "AppleExerciseTime"}
    mean_types = {"RestingHeartRate", "WalkingHeartRateAverage", "HeartRateVariabilitySDNN"}
    max_types = {"VO2Max"}
    target = sum_types | mean_types | max_types

    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    maxes: dict[str, dict[str, float]] = {}

    ctx = etree.iterparse(xml_path, events=("end",), tag=("Record",))
    for _, elem in ctx:
        a = elem.attrib
        rtype = (
            a.get("type", "")
            .replace("HKQuantityTypeIdentifier", "")
            .replace("HKCategoryTypeIdentifier", "")
        )
        if rtype not in target:
            elem.clear()
            continue
        val = pd.to_numeric(a.get("value"), errors="coerce")
        if pd.isna(val):
            elem.clear()
            continue
        start = pd.to_datetime(a.get("startDate", a.get("creationDate")), utc=True)
        date = start.tz_convert(tz).date().isoformat()

        if rtype in sum_types:
            # unit normalization for energies
            if rtype in {"ActiveEnergyBurned", "BasalEnergyBurned"}:
                val = to_kcal(val, a.get("unit")) or 0.0
            bucket = sums.setdefault(date, {})
            bucket[rtype] = float(bucket.get(rtype, 0.0) + float(val))
        elif rtype in mean_types:
            bucket = sums.setdefault(date, {})
            cb = counts.setdefault(date, {})
            bucket[rtype] = float(bucket.get(rtype, 0.0) + float(val))
            cb[rtype] = int(cb.get(rtype, 0) + 1)
        elif rtype in max_types:
            bucket = maxes.setdefault(date, {})
            bucket[rtype] = float(max(float(val), float(bucket.get(rtype, float("-inf")))))
        elem.clear()

    # Assemble rows
    dates = sorted(set(sums.keys()) | set(counts.keys()) | set(maxes.keys()))
    rows: list[dict] = []
    for d in dates:
        s = sums.get(d, {})
        c = counts.get(d, {})
        m = maxes.get(d, {})

        def avg(key: str, sums: dict[str, float], counts: dict[str, int]) -> float:
            if counts.get(key):
                return float(sums.get(key, 0.0)) / float(counts.get(key, 1))
            return 0.0

        row = {
            "date": d,
            "steps": float(s.get("StepCount", 0.0)),
            "active_kcal": float(s.get("ActiveEnergyBurned", 0.0)),
            "basal_kcal": float(s.get("BasalEnergyBurned", 0.0)),
            "exercise_min": float(s.get("AppleExerciseTime", 0.0)),
            "rhr_bpm": avg("RestingHeartRate", s, c),
            "walking_hr_avg_bpm": avg("WalkingHeartRateAverage", s, c),
            "hrv_sdnn_ms": avg("HeartRateVariabilitySDNN", s, c),
            "vo2max_mlkgmin": float(m.get("VO2Max", 0.0)),
        }
        rows.append(row)
    return pd.DataFrame(rows)
