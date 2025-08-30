from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd


def to_local(ts: pd.Series | pd.DatetimeIndex, tz_name: str) -> pd.Series:
    tz = ZoneInfo(tz_name)
    s = pd.to_datetime(ts, utc=True)
    return s.dt.tz_convert(tz)
