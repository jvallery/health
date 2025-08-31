from __future__ import annotations

import pandas as pd

# Reuse existing computations
from analysis.trends import (
    efficiency_trend,
    monthly_perf,
    weekly_volume,
    yearly_perf,
)


def running_monthly(workouts: pd.DataFrame) -> pd.DataFrame:
    return monthly_perf(workouts)


def running_weekly(workouts: pd.DataFrame) -> pd.DataFrame:
    return weekly_volume(workouts)


def running_yearly(workouts: pd.DataFrame) -> pd.DataFrame:
    return yearly_perf(workouts)


def running_efficiency(workouts: pd.DataFrame) -> pd.DataFrame:
    return efficiency_trend(workouts)

