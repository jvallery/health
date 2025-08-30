from __future__ import annotations

import hashlib

import pandas as pd
from lxml import etree

from ..utils import units


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def parse_workouts(xml_path: str) -> pd.DataFrame:
    rows: list[dict] = []
    ctx = etree.iterparse(xml_path, events=("end",), tag=("Workout",))
    for _, elem in ctx:
        a = elem.attrib
        wtype = a.get("workoutActivityType", "").replace("HKWorkoutActivityType", "")
        # Parse original to capture offset, then convert to UTC for storage
        start_orig = pd.to_datetime(a["startDate"], utc=False)
        start = pd.to_datetime(a["startDate"], utc=True)
        end = pd.to_datetime(a["endDate"], utc=True)
        dur_sec = float(a.get("duration", (end - start).total_seconds() / 60.0)) * 60.0
        dist, dist_u = a.get("totalDistance"), a.get("totalDistanceUnit")
        kcal, kcal_u = a.get("totalEnergyBurned"), a.get("totalEnergyBurnedUnit")

        # Child metadata lookup
        meta = {m.attrib.get("key"): m.attrib.get("value") for m in elem.findall("MetadataEntry")}
        avg_hr = float(meta["HKAverageHeartRate"]) if "HKAverageHeartRate" in meta else None
        max_hr = float(meta["HKMaximumHeartRate"]) if "HKMaximumHeartRate" in meta else None
        min_hr = float(meta["HKMinimumHeartRate"]) if "HKMinimumHeartRate" in meta else None

        tz_offset_min = (
            int(start_orig.utcoffset().total_seconds() / 60)
            if start_orig.tzinfo and start_orig.utcoffset() is not None
            else None
        )

        row = {
            "workout_id": None,  # filled below
            "activity": wtype,
            "source_name": a.get("sourceName"),
            "device": a.get("device"),
            "start_utc": start,
            "end_utc": end,
            "tz_offset_min": tz_offset_min,
            "duration_sec": dur_sec,
            "distance_m": units.to_meters(dist, dist_u),
            "energy_kcal": units.to_kcal(kcal, kcal_u),
            "avg_hr_bpm": avg_hr,
            "max_hr_bpm": max_hr,
            "min_hr_bpm": min_hr,
            "has_route": False,
            "route_id": None,
        }
        key = "|".join(
            [
                str(row["activity"]),
                str(row["start_utc"]),
                str(row["duration_sec"]),
                str(row["distance_m"]),
                str(row["energy_kcal"]),
            ]
        )
        row["workout_id"] = _sha1(key)
        rows.append(row)
        elem.clear()
    return pd.DataFrame(rows)


def parse_records(xml_path: str, types: list[str]) -> dict[str, pd.DataFrame]:
    target = set(types)
    buckets: dict[str, list[dict]] = {t: [] for t in target}
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
        end = pd.to_datetime(a.get("endDate", a.get("creationDate")), utc=True)
        buckets[rtype].append(
            {
                "type": rtype,
                "start_utc": start,
                "end_utc": end,
                "value": a.get("value"),
                "unit": a.get("unit"),
                "source_name": a.get("sourceName"),
            }
        )
        elem.clear()
    return {k: pd.DataFrame(v) for k, v in buckets.items()}
