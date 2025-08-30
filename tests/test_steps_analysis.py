from __future__ import annotations

import pandas as pd
from pathlib import Path

from steps.common import StepsConfig, normalize_daily
from steps.tables import monthly_rollup, weekly_rollup, goal_streaks
from steps.plots import _calendar_matrix_for_year


def make_daily(start: str = "2024-01-01", days: int = 10, steps: int = 1000) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"date": dates, "steps": steps})


def test_normalize_and_miles():
    df = make_daily()
    cfg = StepsConfig(stride_m=0.8)
    d = normalize_daily(df, cfg)
    assert {"date", "steps", "dow", "year", "month", "week_start", "goal_hit", "miles_est"}.issubset(d.columns)
    assert (d["steps"] >= 0).all()
    assert abs(d["miles_est"].iloc[0] - (1000 * 0.8 / 1609.34)) < 1e-6


def test_mass_balance_month_week():
    df = make_daily(days=30, steps=2000)
    d = normalize_daily(df, StepsConfig())
    m = monthly_rollup(d)
    w = weekly_rollup(d)
    assert abs(m["total_steps"].sum() - d["steps"].sum()) < 1e-6
    assert abs(w["total_steps"].sum() - d["steps"].sum()) < 1e-6


def test_streaks():
    # Build pattern: 3 goal days, 1 break, 2 goal days
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    steps = [12000, 11000, 10000, 0, 13000, 14000]
    d = pd.DataFrame({"date": dates, "steps": steps})
    d = normalize_daily(d, StepsConfig(step_goal=10000))
    s = goal_streaks(d, StepsConfig(step_goal=10000)).iloc[0]
    assert int(s["best_streak_days"]) >= int(s["current_streak_days"]) >= 0


def test_calendar_sum():
    # 2024 not leap year
    d = make_daily("2024-01-01", days=365, steps=1000)
    d = normalize_daily(d, StepsConfig())
    mat = _calendar_matrix_for_year(d, 2024)
    assert int(mat.sum()) == 365000

