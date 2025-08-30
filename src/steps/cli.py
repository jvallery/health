from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from .common import StepsConfig, load_daily, normalize_daily
from . import tables as T
from . import plots as P
from .report import write_markdown_report


@click.command()
@click.option("--in", "in_csv", required=True, help="Path to daily_summary.csv")
@click.option("--out", "outdir", required=True, help="Output dir (e.g., out/steps)")
@click.option("--goal", "step_goal", default=10_000, show_default=True)
@click.option("--stride", "stride_m", default=0.78, show_default=True)
@click.option("--start", default=None, help="Filter start date YYYY-MM-DD")
@click.option("--end", default=None, help="Filter end date YYYY-MM-DD")
def analyze(in_csv: str, outdir: str, step_goal: int, stride_m: float, start: str | None, end: str | None) -> None:
    # Config & IO
    if step_goal <= 0:
        click.echo("[WARN] step_goal <= 0; resetting to 10,000")
        step_goal = 10_000
    if stride_m <= 0:
        click.echo("[WARN] stride_m <= 0; resetting to 0.78")
        stride_m = 0.78
    cfg = StepsConfig(step_goal=step_goal, stride_m=stride_m, start=start, end=end)
    out = Path(outdir)
    tdir = out / "tables"
    pdir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    # 1) Load & normalize
    daily_raw = load_daily(in_csv)
    if "steps" not in daily_raw.columns:
        raise SystemExit("ERROR: steps column required.")
    daily = normalize_daily(daily_raw, cfg)

    # 2) Tables
    ls = T.lifetime_summary(daily, cfg)
    ls.to_csv(tdir / "steps_lifetime_summary.csv", index=False)
    click.echo("[OK] lifetime summary → tables/steps_lifetime_summary.csv")

    monthly = T.monthly_rollup(daily)
    monthly.to_csv(tdir / "steps_monthly.csv", index=False)
    click.echo("[OK] monthly rollup → tables/steps_monthly.csv")

    yearly = T.yearly_rollup(daily)
    yearly.to_csv(tdir / "steps_yearly.csv", index=False)
    click.echo("[OK] yearly rollup → tables/steps_yearly.csv")

    weekly = T.weekly_rollup(daily)
    weekly.to_csv(tdir / "steps_weekly.csv", index=False)
    click.echo("[OK] weekly rollup → tables/steps_weekly.csv")

    dist = T.steps_distribution(daily)
    dist.to_csv(tdir / "steps_distribution.csv", index=False)
    click.echo("[OK] distribution → tables/steps_distribution.csv")

    dow = T.dow_stats(daily)
    dow.to_csv(tdir / "steps_dayofweek.csv", index=False)
    click.echo("[OK] dow stats → tables/steps_dayofweek.csv")

    top, low = T.top_low_days(daily)
    top.to_csv(tdir / "steps_top_days.csv", index=False)
    low.to_csv(tdir / "steps_low_days.csv", index=False)
    click.echo("[OK] top/low days → tables/steps_top_days.csv, steps_low_days.csv")

    streaks = T.goal_streaks(daily, cfg)
    streaks.to_csv(tdir / "steps_streaks.csv", index=False)
    click.echo("[OK] streaks → tables/steps_streaks.csv")

    corr = T.steps_correlations(daily)
    if corr is not None and not corr.empty:
        corr.to_csv(tdir / "steps_correlations.csv", index=False)
        click.echo("[OK] correlations → tables/steps_correlations.csv")
    else:
        for v in ["active_kcal", "exercise_min", "rhr_bpm", "vo2max_mlkgmin", "walking_hr_avg_bpm"]:
            if v not in daily:
                click.echo(f"[SKIP] corr({v}) → column missing")

    # Mass balance checks (log only)
    try:
        if abs(monthly["total_steps"].sum() - daily["steps"].sum()) > 1e-6:
            click.echo("[WARN] monthly sum != daily sum")
        if abs(weekly["total_steps"].sum() - daily["steps"].sum()) > 1e-6:
            click.echo("[WARN] weekly sum != daily sum")
        if dist["days"].sum() != len(daily):
            click.echo("[WARN] bucket days != daily rows")
    except Exception:
        pass

    # 3) Plots
    P.plot_daily_with_rolling(daily, pdir)
    P.plot_weekly_steps(weekly, pdir)
    P.plot_monthly_totals_and_goal_days(monthly, cfg, pdir)
    P.plot_yearly_totals(yearly, pdir)
    P.plot_steps_hist(daily, pdir)
    P.plot_steps_cdf(daily, pdir)
    P.plot_dow_average(dow, pdir)
    P.plot_bucket_distribution(dist, pdir)
    P.plot_weekly_goal_pct(weekly, pdir)
    # Calendar plots with common scale
    years = sorted(daily["date"].dt.year.dropna().astype(int).unique().tolist())
    vmax = float(np.nanpercentile(daily["steps"], 95)) if not daily.empty else None
    for y in years:
        P.plot_calendar_heatmap(daily, y, pdir, vmin=0.0, vmax=vmax)
        P.plot_cumulative_vs_goal(daily, y, cfg, pdir)
    # Optional scatters
    for v in ["active_kcal", "exercise_min", "rhr_bpm", "vo2max_mlkgmin", "walking_hr_avg_bpm"]:
        out_plot = P.plot_steps_vs_var(daily, v, pdir)
        if out_plot:
            click.echo(f"[OK] scatter {v} → {out_plot}")
        else:
            if v not in daily:
                click.echo(f"[SKIP] steps_vs_{v} → column missing")
            else:
                click.echo(f"[SKIP] steps_vs_{v} → insufficient data")

    # 4) Report
    write_markdown_report(daily, monthly, yearly, weekly, dow, dist, streaks, corr, years, cfg, out)
    click.echo(f"[OK] report → {out/'report_steps.md'}")


if __name__ == "__main__":  # pragma: no cover
    analyze()
