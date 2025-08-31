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
    mean_types = {
        "RestingHeartRate",
        "WalkingHeartRateAverage",
        "HeartRateVariabilitySDNN",
        "RespiratoryRate",
        "OxygenSaturation",
    }
    max_types = {"VO2Max"}
    last_types = {"BodyMass", "BodyFatPercentage", "Height", "BloodPressureSystolic", "BloodPressureDiastolic"}
    category_types = {"SleepAnalysis"}
    target = sum_types | mean_types | max_types | last_types | category_types

    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    maxes: dict[str, dict[str, float]] = {}
    lasts: dict[str, dict[str, tuple[pd.Timestamp, float]]] = {}
    sleep: dict[str, dict[str, float]] = {}

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
        start = pd.to_datetime(a.get("startDate", a.get("creationDate")), utc=True)
        end = pd.to_datetime(a.get("endDate", a.get("startDate")), utc=True)
        date = start.tz_convert(tz).date().isoformat()

        if rtype in category_types:
            # SleepAnalysis: accumulate minutes by category
            sval = a.get("value", "")
            sval_str = str(sval)
            dur_min = float((end - start).total_seconds() / 60.0)
            bucket = sleep.setdefault(date, {"sleep_asleep_min": 0.0, "sleep_in_bed_min": 0.0})
            text = sval_str.lower()
            # map various encodings
            is_asleep = any(k in text for k in ["asleep", "core", "deep", "rem"]) or (str(sval).isdigit() and int(str(sval)) >= 1)
            is_inbed = ("inbed" in text) or (str(sval).isdigit() and int(str(sval)) == 0)
            if is_asleep:
                bucket["sleep_asleep_min"] += dur_min
            if is_inbed:
                bucket["sleep_in_bed_min"] += dur_min
            # if neither flagged, ignore (awake segments)
        else:
            val = pd.to_numeric(a.get("value"), errors="coerce")
            if pd.isna(val):
                elem.clear()
                continue
            if rtype in sum_types:
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
            elif rtype in last_types:
                b = lasts.setdefault(date, {})
                b[rtype] = (start, float(val))
        elem.clear()

    # Assemble rows
    dates = sorted(set(sums.keys()) | set(counts.keys()) | set(maxes.keys()) | set(lasts.keys()) | set(sleep.keys()))
    rows: list[dict] = []
    for d in dates:
        s = sums.get(d, {})
        c = counts.get(d, {})
        m = maxes.get(d, {})
        l = lasts.get(d, {})
        sl = sleep.get(d, {})

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
            # Body & vitals
            "weight_kg": float(l.get("BodyMass", (None, None))[1] or 0.0),
            "body_fat_pct": float(l.get("BodyFatPercentage", (None, None))[1] or 0.0),
            "height_cm": float((l.get("Height", (None, None))[1] or 0.0) * (100.0 if l.get("Height") else 1.0)),
            "bp_systolic": float(l.get("BloodPressureSystolic", (None, None))[1] or 0.0),
            "bp_diastolic": float(l.get("BloodPressureDiastolic", (None, None))[1] or 0.0),
            "respiratory_rate_bpm": avg("RespiratoryRate", s, c),
            "spo2_pct": avg("OxygenSaturation", s, c),
            # Sleep
            **({} if not sl else {"sleep_asleep_min": float(sl.get("sleep_asleep_min", 0.0)), "sleep_in_bed_min": float(sl.get("sleep_in_bed_min", 0.0))}),
        }
        rows.append(row)
    return pd.DataFrame(rows)
