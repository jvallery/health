from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class StepsConfig:
    step_goal: int = 10_000
    stride_m: float = 0.78
    user_tz: str = "America/Denver"
    start: str | None = None
    end: str | None = None


def load_daily(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def normalize_daily(df: pd.DataFrame, cfg: StepsConfig) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        raise ValueError("steps column required.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date").reset_index(drop=True)
    # Steps
    if "steps" not in d.columns:
        raise SystemExit("ERROR: steps column required.")
    d["steps"] = pd.to_numeric(d["steps"], errors="coerce").fillna(0).clip(lower=0)
    # Derived
    d["dow"] = d["date"].dt.day_name()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.to_period("M").astype(str)
    # Week start (Monday). Using W-MON.start_time returns Tuesday; compute directly.
    d["week_start"] = d["date"].dt.normalize() - pd.to_timedelta(d["date"].dt.weekday, unit="D")
    d["goal_hit"] = d["steps"] >= max(1, int(cfg.step_goal))
    d["miles_est"] = pd.to_numeric(d["steps"], errors="coerce") * float(cfg.stride_m) / 1609.34

    # Optional filter
    if cfg.start:
        d = d[d["date"] >= pd.to_datetime(cfg.start)]
    if cfg.end:
        d = d[d["date"] <= pd.to_datetime(cfg.end)]
    return d
