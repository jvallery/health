from __future__ import annotations

import pandas as pd

from analysis.intensity import estimate_hrmax, estimate_hrrest, zones_trimp_for_workouts
from analysis.outliers import efficiency_residuals, multivariate_outliers, pace_zscores_by_bucket
from analysis.prs import pr_by_distance_bucket, pr_common_races
from analysis.load import weekly_load, acute_chronic_ratio


def compute_zones_trimp(
    workouts: pd.DataFrame, hr_samples: pd.DataFrame, daily_summary: pd.DataFrame | None
) -> tuple[pd.DataFrame, float | None, float | None]:
    if hr_samples is None or hr_samples.empty:
        return pd.DataFrame(), None, None
    hrmax = estimate_hrmax(workouts, hr_samples)
    hrrest = estimate_hrrest(daily_summary if daily_summary is not None else pd.DataFrame(), hr_samples)
    zt = zones_trimp_for_workouts(hr_samples, workouts, hrmax, hrrest)
    return zt, hrmax, hrrest


def compute_outliers(workouts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = pace_zscores_by_bucket(workouts)
    res = efficiency_residuals(workouts)
    maha = multivariate_outliers(workouts)
    return z, res, maha


def compute_prs(workouts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pr_by_distance_bucket(workouts), pr_common_races(workouts)


def compute_load(wk_with_trimp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wl = weekly_load(wk_with_trimp)
    ac = acute_chronic_ratio(wl) if not wl.empty else pd.DataFrame()
    return wl, ac

