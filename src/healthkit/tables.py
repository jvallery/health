from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is not None and not df.empty:
        df.to_csv(path, index=False)


def write_running_tables(
    out_base: Path,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    yearly: pd.DataFrame,
    prs: pd.DataFrame,
    prs_races: pd.DataFrame,
    outliers_z: pd.DataFrame,
    efficiency_resid: pd.DataFrame,
    maha: pd.DataFrame,
    wl: pd.DataFrame,
    ac: pd.DataFrame,
) -> None:
    tdir = out_base / "tables" / "running"
    _write(monthly, tdir / "run_monthly_perf.csv")
    _write(weekly, tdir / "run_weekly_mileage.csv")
    _write(yearly, tdir / "run_yearly_perf.csv")
    _write(prs, tdir / "run_prs_by_bucket.csv")
    _write(prs_races, tdir / "run_prs_common_races.csv")
    _write(outliers_z, tdir / "run_outliers_pace_z.csv")
    _write(efficiency_resid, tdir / "run_efficiency_residuals.csv")
    _write(maha, tdir / "run_multivariate_outliers.csv")
    _write(wl, tdir / "run_weekly_trimp.csv")
    _write(ac, tdir / "run_acute_chronic.csv")


def write_steps_tables(out_base: Path, tables: dict[str, pd.DataFrame]) -> None:
    tdir = out_base / "tables" / "steps"
    for name, df in tables.items():
        _write(df, tdir / f"{name}.csv")


def write_vitals_tables(out_base: Path, daily_rollups: pd.DataFrame) -> None:
    tdir = out_base / "tables" / "vitals"
    _write(daily_rollups, tdir / "vitals_daily_rollups.csv")


def write_environment_tables(out_base: Path, joined: pd.DataFrame, effects: pd.DataFrame) -> None:
    tdir = out_base / "tables" / "environment"
    _write(joined, tdir / "workouts_weather_joined.csv")
    _write(effects, tdir / "weather_effects.csv")


def write_alerts_tables(out_base: Path, ecg_counts: pd.DataFrame) -> None:
    tdir = out_base / "tables" / "alerts"
    _write(ecg_counts, tdir / "ecg_classification_counts.csv")

