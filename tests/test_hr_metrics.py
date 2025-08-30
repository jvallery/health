import pandas as pd

from healthkit.transform.hr_metrics import estimate_hrmax, zones_and_load_for_all


def test_zones_and_load_simple():
    # One 4-minute workout, HR ramps from 90 to 150
    start = pd.Timestamp("2025-01-01T12:00:00Z")
    end = start + pd.Timedelta(minutes=4)
    workouts = pd.DataFrame(
        [
            {
                "workout_id": "w1",
                "start_utc": start,
                "end_utc": end,
                "max_hr_bpm": 160,
            }
        ]
    )
    ts = pd.date_range(start, end, freq="5s", inclusive="left")
    bpm = pd.Series(90 + (ts - start).total_seconds() / (4 * 60) * 60)
    hr = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "bpm": bpm,
            "workout_id": "w1",
        }
    )

    hrmax = estimate_hrmax(workouts, hr)
    zones = zones_and_load_for_all(workouts, hr, hrmax)
    assert not zones.empty
    assert {"z1_min", "z2_min", "z3_min", "z4_min", "z5_min"}.issubset(zones.columns)
    assert zones.loc[0, "edwards_points"] > 0
