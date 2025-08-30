from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_report(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    weekly: pd.DataFrame,
    dow_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    streaks_df: pd.DataFrame,
    corr_df: pd.DataFrame | None,
    years: list[int],
    cfg,
    out_dir: str | Path,
) -> str:
    out_path = Path(out_dir)
    # Accept both dir path and a file path; ensure directory exists
    out_path.mkdir(parents=True, exist_ok=True)
    p = out_path / "report_steps.md"
    start = daily["date"].min().date().isoformat() if not daily.empty else ""
    end = daily["date"].max().date().isoformat() if not daily.empty else ""
    total = float(daily["steps"].sum()) if not daily.empty else 0.0
    miles = float(daily.get("miles_est", pd.Series(0, index=daily.index)).sum()) if not daily.empty else 0.0
    lines = []
    lines.append(f"# Steps Report ({start} → {end})")
    lines.append("")
    lines.append(f"- Total steps: {int(total):,}")
    lines.append(f"- Estimated miles: {miles:.1f}")
    if streaks_df is not None and not streaks_df.empty:
        s = streaks_df.iloc[0]
        lines.append(f"- Best streak: {int(s['best_streak_days'])} days; Current: {int(s['current_streak_days'])} days (goal {int(s['step_goal']):,})")
    lines.append("")
    # Links to tables
    lines.append("## Tables")
    tdir = "tables"
    for name in [
        "steps_lifetime_summary.csv",
        "steps_monthly.csv",
        "steps_yearly.csv",
        "steps_weekly.csv",
        "steps_dayofweek.csv",
        "steps_distribution.csv",
        "steps_top_days.csv",
        "steps_low_days.csv",
        "steps_streaks.csv",
        "steps_correlations.csv",
    ]:
        path = out_path / tdir / name
        if path.exists():
            lines.append(f"- {name}: ./{tdir}/{name}")
    lines.append("")
    # Plots
    lines.append("## Plots")
    pdir = "plots"
    plot_names = [
        "steps_daily_rolling.png",
        "steps_weekly_line.png",
        "steps_monthly_totals_and_goal_days.png",
        "steps_yearly_totals.png",
        "steps_hist.png",
        "steps_cdf.png",
        "steps_dow_avg.png",
        "steps_bucket_distribution.png",
        "steps_weekly_goal_pct.png",
    ]
    for nm in plot_names:
        f = out_path / pdir / nm
        if f.exists():
            lines.append(f"### {nm}")
            lines.append(f"![{nm}](./{pdir}/{nm})")
    # Calendar steps and cumulative per year
    for y in years:
        c = out_path / pdir / f"calendar_steps_{y}.png"
        if c.exists():
            lines.append(f"### calendar_steps_{y}.png")
            lines.append(f"![calendar_steps_{y}](./{pdir}/calendar_steps_{y}.png)")
        cu = out_path / pdir / f"steps_cumulative_{y}.png"
        if cu.exists():
            lines.append(f"### steps_cumulative_{y}.png")
            lines.append(f"![steps_cumulative_{y}](./{pdir}/steps_cumulative_{y}.png)")
    p.write_text("\n".join(lines))
    return str(p)
