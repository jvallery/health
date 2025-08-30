from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write(df: pd.DataFrame, out_dir: Path, name: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, index=False)
    return df


def lifetime_summary(daily: pd.DataFrame, cfg) -> pd.DataFrame:
    d = daily.copy()
    days = len(d)
    total_steps = float(d["steps"].sum())
    total_miles = float(d.get("miles_est", pd.Series(0.0, index=d.index)).sum())
    avg = float(d["steps"].mean()) if days else 0.0
    med = float(d["steps"].median()) if days else 0.0
    p90 = float(np.nanpercentile(d["steps"], 90)) if days else 0.0
    goal_days = int(d.get("goal_hit", pd.Series(False, index=d.index)).sum())
    goal_rate = float(goal_days) / float(days) if days else 0.0
    best = d.sort_values("steps", ascending=False).iloc[0] if days else None
    worst_nz = d[d["steps"] > 0].sort_values("steps").iloc[0] if (d["steps"] > 0).any() else None
    out = pd.DataFrame(
        [
            {
                "days": int(days),
                "total_steps": total_steps,
                "total_miles_est": total_miles,
                "avg_steps_per_day": avg,
                "median_steps_per_day": med,
                "p90_steps": p90,
                "days_goal_hit": int(goal_days),
                "goal_hit_rate": goal_rate,
                "best_day_date": best["date"].date().isoformat() if best is not None else "",
                "best_day_steps": float(best["steps"]) if best is not None else 0.0,
                "worst_nonzero_date": worst_nz["date"].date().isoformat() if worst_nz is not None else "",
                "worst_nonzero_steps": float(worst_nz["steps"]) if worst_nz is not None else 0.0,
            }
        ]
    )
    return out


def monthly_rollup(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("month", dropna=True).agg(
        days=("steps", "size"),
        total_steps=("steps", "sum"),
        avg_steps=("steps", "mean"),
        median_steps=("steps", "median"),
        goal_days=("goal_hit", "sum"),
        miles_est=("miles_est", "sum"),
    )
    return g.reset_index()


def yearly_rollup(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("year", dropna=True).agg(
        days=("steps", "size"),
        total_steps=("steps", "sum"),
        avg_steps=("steps", "mean"),
        goal_days=("goal_hit", "sum"),
        miles_est=("miles_est", "sum"),
    )
    return g.reset_index()


def weekly_rollup(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("week_start", dropna=True).agg(
        days=("steps", "size"), total_steps=("steps", "sum"), avg_steps=("steps", "mean"), goal_days=("goal_hit", "sum")
    )
    return g.reset_index()


def steps_distribution(daily: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 3000, 5000, 8000, 10000, 12000, 15000, 20000, np.inf]
    labels = ["<3k", "3–5k", "5–8k", "8–10k", "10–12k", "12–15k", "15–20k", "20k+"]
    bucket = pd.cut(daily["steps"], bins=bins, labels=labels, right=False)
    g = bucket.value_counts().reindex(labels, fill_value=0).rename_axis("bucket").rename("days").reset_index()
    return g


def dow_stats(daily: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = (
        daily.groupby("dow").agg(days=("steps", "size"), avg_steps=("steps", "mean"), median_steps=("steps", "median"), goal_days=("goal_hit", "sum"))
    )
    g = g.reindex(order)
    return g.reset_index().rename(columns={"dow": "day"})


def top_low_days(daily: pd.DataFrame, k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = daily[["date", "steps", "miles_est"]].copy()
    top = d.sort_values("steps", ascending=False).head(k)
    low = d[d["steps"] > 0].sort_values("steps").head(k)
    return top, low


def goal_streaks(daily: pd.DataFrame, cfg) -> pd.DataFrame:
    # assumes normalized and sorted by date
    d = daily.copy()
    best = cur = 0
    prev = None
    for _, r in d.iterrows():
        if not r["goal_hit"]:
            cur = 0
            prev = r["date"]
            continue
        if prev is None:
            cur = 1
        else:
            delta = (r["date"] - prev).days
            if delta == 1 and r["goal_hit"]:
                cur += 1
            else:
                cur = 1 if r["goal_hit"] else 0
        best = max(best, cur)
        prev = r["date"]
    out = pd.DataFrame(
        [{"best_streak_days": int(best), "current_streak_days": int(cur), "step_goal": int(cfg.step_goal)}]
    )
    return out


def steps_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    vars = ["active_kcal", "exercise_min", "rhr_bpm", "vo2max_mlkgmin", "walking_hr_avg_bpm"]
    rows = []
    for v in vars:
        if v not in daily:
            continue
        x = pd.to_numeric(daily[v], errors="coerce")
        y = pd.to_numeric(daily["steps"], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        r = float(x[mask].corr(y[mask], method="pearson"))
        rows.append({"variable": v, "pearson_r": r, "n": int(mask.sum())})
    return pd.DataFrame(rows)

