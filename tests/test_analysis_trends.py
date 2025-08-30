import pandas as pd

from analysis.trends import efficiency_trend, monthly_perf


def _mk_sample():
    ts = pd.date_range("2025-01-01", periods=10, freq="7D", tz="UTC")
    return pd.DataFrame(
        {
            "start_local": ts,
            "distance_m": [5000.0] * len(ts),
            "duration_sec": [1800.0 + i * 30 for i in range(len(ts))],
            "avg_hr_bpm": [130 + i % 5 for i in range(len(ts))],
        }
    )


def test_monthly_and_efficiency():
    w = _mk_sample()
    m = monthly_perf(w)
    assert not m.empty
    e = efficiency_trend(w)
    assert not e.empty
