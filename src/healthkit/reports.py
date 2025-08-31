from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _md_table(df_path: Path, max_rows: int = 15) -> str:
    if not df_path.exists():
        return "_No data_"
    df = pd.read_csv(df_path)
    if df.empty:
        return "_No data_"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _nav() -> str:
    return (
        "[Index](./index.md) | "
        "[Running](./running.md) | "
        "[Steps](./steps.md) | "
        "[Body](./body.md) | "
        "[Vitals](./vitals.md) | "
        "[Sleep](./sleep.md) | "
        "[Alerts](./alerts.md)"
    )


def write_master_index(out_base: Path) -> None:
    p = out_base / "reports" / "index.md"
    lines: list[str] = [
        _nav(),
        "",
        "# Health Analytics",
        "",
        "## Overview",
        "",
        "![Running: Miles & Runs](../plots/running/run_monthly_miles_and_runs.png)",
        "",
        "![Steps: Daily with 30D](../plots/steps/steps_daily_rolling.png)",
        "",
        "![Body: Weight](../plots/body/body_weight_trend.png)",
        "",
        "## Sections",
        "- [Running](./running.md)",
        "- [Steps](./steps.md)",
        "- [Body](./body.md)",
        "- [Vitals](./vitals.md)",
        "- [Sleep](./sleep.md)",
        "- [Alerts](./alerts.md)",
    ]
    # Data Summary: activity and source counts
    try:
        def _load_any(base: Path, stem: str) -> pd.DataFrame:
            pq = base / f"{stem}.parquet"; cs = base / f"{stem}.csv"
            if pq.exists():
                return pd.read_parquet(pq)
            if cs.exists():
                return pd.read_csv(cs)
            return pd.DataFrame()

        ws = _load_any(out_base, "workout_stats")
        if ws.empty:
            ws = _load_any(out_base, "workouts")
        if not ws.empty:
            lines += ["", "## Data Summary", ""]
            # Inline charts if present
            p_sum = out_base / "plots" / "summary"
            by_act = p_sum / "summary_workouts_by_activity.png"
            by_src = p_sum / "summary_sources_top10.png"
            if by_act.exists() or by_src.exists():
                lines += ["### Charts", ""]
                if by_act.exists():
                    lines += ["#### Workouts by Activity", f"![by activity](../plots/summary/{by_act.name})", ""]
                if by_src.exists():
                    lines += ["#### Top Sources", f"![top sources](../plots/summary/{by_src.name})", ""]
            d = ws.copy()
            # Distance coverage (prefer GPX when present)
            dist = pd.to_numeric(d.get("gpx_distance_m"), errors="coerce")
            if "distance_m" in d.columns:
                dist = dist.fillna(pd.to_numeric(d["distance_m"], errors="coerce"))
            has_dist = dist.fillna(0) > 0
            has_hr = pd.to_numeric(d.get("avg_hr_bpm"), errors="coerce").notna()
            has_route = d.get("route_id").notna() if "route_id" in d.columns else pd.Series(False, index=d.index)
            # Activity breakdown
            act = (
                d.assign(has_dist=has_dist, has_hr=has_hr, has_route=has_route)
                .groupby(d.get("activity").fillna("Unknown"))
                .agg(total=("has_dist", "size"), with_distance=("has_dist", "sum"), with_hr=("has_hr", "sum"), with_route=("has_route", "sum"))
                .sort_values("total", ascending=False)
                .reset_index()
                .rename(columns={"index": "activity", "activity": "activity"})
            )
            # Source breakdown (top 10)
            src = (
                d.groupby(d.get("source_name").fillna("Unknown"))
                .size()
                .rename("count")
                .sort_values(ascending=False)
                .reset_index()
                .head(10)
                .rename(columns={"index": "source_name", "source_name": "source_name"})
            )
            # Render tables inline
            def _md_table_df(df: pd.DataFrame, max_rows: int = 20) -> str:
                if df is None or df.empty:
                    return "_No data_"
                d2 = df.head(max_rows)
                cols = list(d2.columns)
                lines2 = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                for _, row in d2.iterrows():
                    vals = [str(row[c]) for c in cols]
                    lines2.append("| " + " | ".join(vals) + " |")
                return "\n".join(lines2)

            lines += ["### Workouts by Activity", _md_table_df(act), ""]
            lines += ["### Top Sources", _md_table_df(src), ""]
    except Exception:
        pass
    _write(p, lines)


def _list_pngs(dirpath: Path) -> list[Path]:
    if not dirpath.exists():
        return []
    return sorted([p for p in dirpath.glob("*.png")])


def write_running(out_base: Path) -> None:
    base = out_base / "reports"
    lines: list[str] = [_nav(), "", "# Running", ""]
    rdir = out_base / "plots" / "running"
    pngs = _list_pngs(rdir)
    names = {p.name: p for p in pngs}

    # Priority multi-year summaries at top
    priority = [
        "run_monthly_miles_and_runs.png",
        "run_monthly_median_pace.png",
    ]
    bottom_plots = [
        "run_acute_chronic.png",
        "run_correlation_matrix.png",
        "run_device_bias_bar.png",
        "run_device_bucket_heatmap.png",
        "run_pr_timeline.png",
        "run_weekly_trimp.png",
    ]
    for nm in priority:
        if nm in names:
            lines += [f"## {nm}", f"![{nm}](../plots/running/{nm})", ""]

    # Embed all remaining non-calendar PNGs
    calendars = [
        p
        for p in pngs
        if p.name.startswith("calendar_steps_") or ("calendar_" in p.name)
    ]
    for p in [p for p in pngs if p.name not in priority and p not in calendars and p.name not in bottom_plots]:
        lines += [f"## {p.name}", f"![{p.name}](../plots/running/{p.name})", ""]

    # Calendars (reverse chronological)
    if calendars:
        lines += ["## Calendars", ""]
        import re

        def _yr(p: Path) -> int:
            m = re.search(r"(20\d{2})", p.name)
            return int(m.group(1)) if m else 0

        for p in sorted(calendars, key=_yr, reverse=True):
            lines += [f"### {p.name}", f"![{p.name}](../plots/running/{p.name})", ""]

    # Move selected plots to bottom (after calendars)
    for nm in bottom_plots:
        if nm in names:
            lines += [f"## {nm}", f"![{nm}](../plots/running/{nm})", ""]

    # Tables block
    tdir = out_base / "tables" / "running"
    lines += ["## Tables", ""]
    for f in [
        "run_monthly_perf.csv",
        "run_weekly_mileage.csv",
        "run_yearly_perf.csv",
        "run_prs_by_bucket.csv",
        "run_outliers_pace_z.csv",
        "run_efficiency_residuals.csv",
        "run_multivariate_outliers.csv",
        "run_weekly_trimp.csv",
        "run_acute_chronic.csv",
    ]:
        path = tdir / f
        if path.exists():
            lines += [f"### {f}", _md_table(path), ""]
    _write(base / "running.md", lines)


def write_steps(out_base: Path) -> None:
    base = out_base / "reports"
    lines = [_nav(), "", "# Steps", ""]
    sdir = out_base / "plots" / "steps"
    pngs = _list_pngs(sdir)
    names = {p.name: p for p in pngs}
    priority = [
        "steps_daily_rolling.png",
        "steps_monthly_totals_and_goal_days.png",
        "steps_yearly_totals.png",
    ]
    for nm in priority:
        if nm in names:
            lines += [f"## {nm}", f"![{nm}](../plots/steps/{nm})", ""]
    calendars = [p for p in pngs if p.name.startswith("calendar_steps_")]
    cumulative = [p for p in pngs if p.name.startswith("steps_cumulative_")]
    # Everything else (non-calendar, non-cumulative)
    for p in [p for p in pngs if p.name not in priority and p not in calendars and p not in cumulative]:
        lines += [f"## {p.name}", f"![{p.name}](../plots/steps/{p.name})", ""]
    # Cumulative (reverse chronological)
    if cumulative:
        lines += ["## Cumulative", ""]
        import re
        def _yr(p: Path) -> int:
            m = re.search(r"(20\d{2})", p.name)
            return int(m.group(1)) if m else 0
        for p in sorted(cumulative, key=_yr, reverse=True):
            lines += [f"### {p.name}", f"![{p.name}](../plots/steps/{p.name})", ""]
    if calendars:
        lines += ["## Calendars", ""]
        def _yr(p: Path) -> int:
            import re
            m = re.search(r"(20\d{2})", p.name)
            return int(m.group(1)) if m else 0
        for p in sorted(calendars, key=_yr, reverse=True):
            lines += [f"### {p.name}", f"![{p.name}](../plots/steps/{p.name})", ""]
    _write(base / "steps.md", lines)


def write_vitals(out_base: Path) -> None:
    base = out_base / "reports"
    lines = [_nav(), "", "# Vitals", ""]
    vdir = out_base / "plots" / "vitals"
    pngs = _list_pngs(vdir)
    names = {p.name: p for p in pngs}
    priority = ["vitals_rhr_trend.png", "vitals_vo2_trend.png", "vitals_walking_hr_avg_trend.png"]
    for nm in priority:
        if nm in names:
            lines += [f"## {nm}", f"![{nm}](../plots/vitals/{nm})", ""]
    for p in [p for p in pngs if p.name not in priority]:
        lines += [f"## {p.name}", f"![{p.name}](../plots/vitals/{p.name})", ""]
    _write(base / "vitals.md", lines)


def write_body(out_base: Path) -> None:
    base = out_base / "reports"
    lines = [_nav(), "", "# Body & Composition", ""]
    bdir = out_base / "plots" / "body"
    pngs = _list_pngs(bdir)
    names = {p.name: p for p in pngs}
    priority = ["body_weight_trend.png", "body_fat_trend.png", "body_bmi_trend.png"]
    for nm in priority:
        if nm in names:
            lines += [f"## {nm}", f"![{nm}](../plots/body/{nm})", ""]
    for p in [p for p in pngs if p.name not in priority and p.name != "body_weight_monthly_summary.png"]:
        lines += [f"## {p.name}", f"![{p.name}](../plots/body/{p.name})", ""]
    _write(base / "body.md", lines)


def write_sleep(out_base: Path) -> None:
    base = out_base / "reports"
    lines = [_nav(), "", "# Sleep", ""]
    for p in _list_pngs(out_base / "plots" / "sleep"):
        rel = f"../plots/sleep/{p.name}"
        lines += [f"## {p.name}", f"![{p.name}]({rel})", ""]
    _write(base / "sleep.md", lines)


def write_alerts(out_base: Path) -> None:
    base = out_base / "reports"
    lines = [_nav(), "", "# ECG, Alerts & Exposure", ""]
    for p in _list_pngs(out_base / "plots" / "alerts"):
        rel = f"../plots/alerts/{p.name}"
        lines += [f"## {p.name}", f"![{p.name}]({rel})", ""]
    _write(base / "alerts.md", lines)


def write_environment(out_base: Path) -> None:
    base = out_base / "reports"
    lines = [_nav(), "", "# Environment", ""]
    for p in _list_pngs(out_base / "plots" / "environment"):
        rel = f"../plots/environment/{p.name}"
        lines += [f"## {p.name}", f"![{p.name}]({rel})", ""]
    _write(base / "environment.md", lines)
