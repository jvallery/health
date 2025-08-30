from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Shared palette from analysis.viz
BASE_BLUE = "#1f5aa6"
BLUE_LIGHT = "#4f97d7"
BLUE_DARK = "#0b3b75"
GREY_MED = "#6e7f8d"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#2b3e50",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dfe7ef",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
    }
)


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_daily_with_rolling(daily: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    d = daily.sort_values("date").copy()
    s = d["steps"].where(d["steps"] > 0, np.nan)
    ax.plot(d["date"], s, color=BASE_BLUE, linewidth=1.2, label="daily", zorder=3)
    r7 = s.rolling(7, min_periods=3).mean()
    r30 = s.rolling(30, min_periods=7).mean()
    ax.plot(d["date"], r7, color=BLUE_LIGHT, linewidth=1.6, label="7‑day avg", zorder=2)
    ax.plot(d["date"], r30, color=GREY_MED, linewidth=1.8, label="30‑day avg", zorder=2)
    ax.set_title("Daily Steps with Rolling Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Steps")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "steps_daily_rolling.png")


def plot_weekly_steps(weekly_df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    y = weekly_df["total_steps"].where(weekly_df["total_steps"] > 0, np.nan)
    ax.plot(weekly_df["week_start"], y, color=BASE_BLUE, linewidth=1.8)
    ax.set_title("Weekly Total Steps")
    ax.set_xlabel("Week start")
    ax.set_ylabel("Steps")
    return _save(fig, out_dir / "steps_weekly_line.png")


def plot_monthly_totals_and_goal_days(monthly_df: pd.DataFrame, cfg, out_dir: Path) -> str:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(monthly_df["month"], monthly_df["total_steps"], color=BASE_BLUE, alpha=0.9, zorder=3)
    ax1.set_ylabel("Total steps", color=BASE_BLUE)
    ax2 = ax1.twinx()
    ax2.plot(monthly_df["month"], monthly_df["goal_days"], color=GREY_MED, linewidth=1.6, marker="o", label="Goal days")
    ax2.set_ylabel("Goal days", color=GREY_MED)
    ax1.set_title(f"Monthly Steps and Goal Days (goal={cfg.step_goal:,})")
    # Rotate + downsample ticks for long timelines
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")
    if len(monthly_df) > 18:
        ticks = list(range(0, len(monthly_df), max(1, len(monthly_df) // 12)))
        ax1.set_xticks([monthly_df["month"].iloc[i] for i in ticks])
    return _save(fig, out_dir / "steps_monthly_totals_and_goal_days.png")


def plot_yearly_totals(yearly_df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly_df["year"].astype(str), yearly_df["total_steps"], color=BASE_BLUE, zorder=3)
    ax.set_title("Yearly Total Steps")
    ax.set_xlabel("Year")
    ax.set_ylabel("Steps")
    return _save(fig, out_dir / "steps_yearly_totals.png")


def plot_steps_hist(daily: pd.DataFrame, out_dir: Path, bins: int = 40) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(daily["steps"].dropna(), bins=bins, color=BASE_BLUE, edgecolor=None, zorder=3)
    ax.set_title("Daily Steps Distribution")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    return _save(fig, out_dir / "steps_hist.png")


def plot_steps_cdf(daily: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.sort(daily["steps"].dropna().to_numpy())
    y = np.linspace(0, 1, len(x), endpoint=True)
    ax.plot(x, y, color=BASE_BLUE, linewidth=1.6)
    ax.set_title("CDF of Daily Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative fraction")
    # Explain CDF inline for clarity
    ax.text(0.01, 0.05, "At value v, the curve shows the fraction of days with steps ≤ v.", transform=ax.transAxes, fontsize=8, color=GREY_MED)
    return _save(fig, out_dir / "steps_cdf.png")


def plot_dow_average(dow_df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow_df["day"], dow_df["avg_steps"], color=BASE_BLUE, zorder=3)
    ax.set_title("Average Steps by Weekday")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Average steps")
    return _save(fig, out_dir / "steps_dow_avg.png")


def plot_bucket_distribution(dist_df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dist_df["bucket"], dist_df["days"], color=BASE_BLUE, zorder=3)
    ax.set_title("Daily Steps Bucket Distribution")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Days")
    return _save(fig, out_dir / "steps_bucket_distribution.png")


def _calendar_matrix_for_year(daily: pd.DataFrame, year: int) -> np.ndarray:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])  # ensure datetime
    dy = d[d["date"].dt.year == year]
    vals = dy.groupby(dy["date"].dt.date)["steps"].sum()
    import datetime as _dt

    start = _dt.date(year, 1, 1)
    start -= _dt.timedelta(days=start.weekday())
    weeks = 54
    mat = np.zeros((7, weeks), dtype=float)
    for w in range(weeks):
        for dow in range(7):
            cur = start + _dt.timedelta(days=w * 7 + dow)
            mat[dow, w] = float(vals.get(cur, 0.0))
    return mat


def plot_calendar_heatmap(daily: pd.DataFrame, year: int, out_dir: Path, vmin: float | None, vmax: float | None) -> str:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    mat = _calendar_matrix_for_year(daily, year)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks([])
    ax.set_title(f"{year} Steps Calendar")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Steps")
    return _save(fig, out_dir / f"calendar_steps_{year}.png")


def plot_weekly_goal_pct(weekly_df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    pct = weekly_df["goal_days"] / weekly_df["days"].replace(0, np.nan)
    ax.plot(weekly_df["week_start"], pct, color=BASE_BLUE, linewidth=1.0, alpha=0.6, label="weekly")
    # 4-week rolling average for readability
    r4 = pct.rolling(4, min_periods=2).mean()
    ax.plot(weekly_df["week_start"], r4, color=GREY_MED, linewidth=1.8, label="4‑wk avg")
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=BLUE_DARK, linewidth=0.8, alpha=0.4)
    ax.set_title("Weekly Goal Adherence")
    ax.set_xlabel("Week start")
    ax.set_ylabel("Goal %")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "steps_weekly_goal_pct.png")


def plot_steps_vs_var(daily: pd.DataFrame, var: str, out_dir: Path) -> str | None:
    if var not in daily:
        return None
    x = pd.to_numeric(daily[var], errors="coerce")
    y = pd.to_numeric(daily["steps"], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x[mask], y[mask], s=12, alpha=0.6, color=BASE_BLUE)
    ax.set_title(f"Steps vs {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Steps")
    # trendline
    try:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        xx = np.linspace(float(x[mask].min()), float(x[mask].max()), 100)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, color=BLUE_DARK, linewidth=1.2)
    except Exception:
        pass
    return _save(fig, out_dir / f"steps_vs_{var}.png")


def plot_cumulative_vs_goal(daily: pd.DataFrame, year: int, cfg, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    d = daily.copy()
    d = d[d["date"].dt.year == year]
    d = d.sort_values("date")
    s = d["steps"].fillna(0)
    cum = s.cumsum()
    days = np.arange(1, len(d) + 1)
    goal_line = days * int(cfg.step_goal)
    ax.plot(d["date"], cum, color=BASE_BLUE, linewidth=1.8, label="Cumulative steps")
    ax.plot(d["date"], goal_line, color=GREY_MED, linestyle="--", linewidth=1.4, label=f"Goal ({int(cfg.step_goal):,}/day)")
    ax.set_title(f"Cumulative Steps vs Goal ({year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Steps")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / f"steps_cumulative_{year}.png")
