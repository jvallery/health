import pandas as pd

from analysis.common import add_effective_fields


def test_add_effective_fields_basic():
    df = pd.DataFrame(
        {
            "duration_sec": [3600, 1800],
            "distance_m": [10000.0, 4000.0],
        }
    )
    out = add_effective_fields(df)
    assert out["distance_m_eff"].notna().mean() == 1.0
    assert out["moving_time_sec_eff"].notna().mean() == 1.0
    assert out["miles"].gt(0).all()
    assert out["pace_min_per_mile_eff"].notna().all()
