from __future__ import annotations

import pandas as pd
from lxml import etree


def _hour_key(ts: pd.Timestamp) -> str:
    return ts.floor("h").isoformat()


def _window_hours(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    cur = start.floor("h")
    hours = []
    while cur <= end:
        hours.append(cur.isoformat())
        cur = cur + pd.Timedelta(hours=1)
    return hours


def hr_samples_near_workouts(
    xml_path: str,
    workouts: pd.DataFrame,
    before: str = "10min",
    after: str = "10min",
) -> pd.DataFrame:
    if workouts.empty:
        return pd.DataFrame(columns=["timestamp_utc", "bpm", "source", "workout_id"])
    w = workouts.copy()
    w["start_utc"] = pd.to_datetime(w["start_utc"], utc=True)
    w["end_utc"] = pd.to_datetime(w["end_utc"], utc=True)
    before_td = pd.Timedelta(before)
    after_td = pd.Timedelta(after)

    # Build hour-bucketed index of windows to reduce comparisons
    buckets: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    for _, row in w.iterrows():
        s = row["start_utc"] - before_td
        e = row["end_utc"] + after_td
        for hk in _window_hours(s, e):
            buckets.setdefault(hk, []).append((s, e, row["workout_id"]))

    rows: list[dict] = []
    ctx = etree.iterparse(xml_path, events=("end",), tag=("Record",))
    for _, elem in ctx:
        a = elem.attrib
        rtype = a.get("type", "").replace("HKQuantityTypeIdentifier", "")
        if rtype != "HeartRate":
            elem.clear()
            continue
        ts = pd.to_datetime(a.get("startDate", a.get("endDate")), utc=True)
        hk = _hour_key(ts)
        candidates = buckets.get(hk)
        if not candidates:
            elem.clear()
            continue
        for s, e, wid in candidates:
            if s <= ts <= e:
                rows.append(
                    {
                        "timestamp_utc": ts,
                        "bpm": float(a.get("value")) if a.get("value") is not None else None,
                        "source": a.get("sourceName"),
                        "workout_id": wid,
                    }
                )
                break
        elem.clear()

    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "bpm", "source", "workout_id"])
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["bpm"]).reset_index(drop=True)
    return df
