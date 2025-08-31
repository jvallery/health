from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from .parsing.ecg_csv import parse_ecg_csv
from .parsing.export_xml import parse_workouts
from .parsing.gpx import parse_gpx_dir
from .transform.daily import daily_summary_stream
from .transform.qa import build_qa_flags
from .transform.routes import link_routes
from .transform.workouts import derive_workout_metrics
from .utils import io as io_utils
from .utils import time as time_utils

console = Console()


@click.group()
def main() -> None:
    """Apple Health analytics CLI."""


@main.command()
@click.option(
    "--xml",
    "xml_path",
    type=click.Path(exists=True, dir_okay=False),
    default="apple_health_export/export.xml",
    show_default=True,
)
@click.option(
    "--routes",
    "routes_dir",
    type=click.Path(exists=True, file_okay=False),
    default="apple_health_export/workout-routes",
    show_default=True,
)
@click.option(
    "--ecg",
    "ecg_dir",
    type=click.Path(exists=True, file_okay=False),
    default="apple_health_export/electrocardiograms",
    show_default=True,
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default="out",
    show_default=True,
)
@click.option("--tz", "tz_name", type=str, default="America/Denver", show_default=True)
@click.option(
    "--with-points/--no-points",
    default=False,
    show_default=True,
    help="Emit route_points parquet",
)
@click.option(
    "--hr-samples/--no-hr-samples",
    default=False,
    show_default=True,
    help="Write hr_samples.parquet (can be large)",
)
def convert(
    xml_path: str,
    routes_dir: str,
    ecg_dir: str,
    out_dir: str,
    tz_name: str,
    with_points: bool,
    hr_samples: bool,
) -> None:
    """Convert Apple Health export to normalized Parquet/CSV bundle."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.rule("Parsing Workouts")
    workouts = parse_workouts(xml_path)
    # Local timestamps
    workouts["start_local"] = time_utils.to_local(workouts["start_utc"], tz_name)
    workouts["end_local"] = time_utils.to_local(workouts["end_utc"], tz_name)
    io_utils.write_parquet_csv(workouts, out / "workouts")
    console.print(f"Workouts: {len(workouts)} → {out/'workouts.parquet'}")

    console.rule("Parsing GPX routes")
    routes_df, points_df = parse_gpx_dir(routes_dir, with_points=with_points)
    if not routes_df.empty:
        io_utils.write_parquet_csv(routes_df, out / "routes")
        console.print(f"Routes: {len(routes_df)} → {out/'routes.parquet'}")
    if with_points and points_df is not None and not points_df.empty:
        io_utils.write_parquet_csv(points_df, out / "route_points")

    console.rule("Linking routes ↔ workouts")
    if not routes_df.empty and not workouts.empty:
        linked = link_routes(workouts.copy(), routes_df.copy())
        io_utils.write_parquet_csv(linked, out / "workouts_linked")
        # Overwrite canonical workouts with link columns
        io_utils.write_parquet_csv(linked, out / "workouts")
        workouts = linked

    console.rule("Deriving workout metrics")
    stats = derive_workout_metrics(workouts, routes_df if not routes_df.empty else None)
    io_utils.write_parquet_csv(stats, out / "workout_stats")

    console.rule("Daily summaries (Records)")
    # Streamed aggregation to avoid loading all Records into memory
    daily = daily_summary_stream(xml_path, tz_name)
    if not daily.empty:
        io_utils.write_parquet_csv(daily, out / "daily_summary")
        console.print(f"Daily rows: {len(daily)} → {out/'daily_summary.parquet'}")

    console.rule("ECG summaries")
    ecg_rows = []
    ecg_path = Path(ecg_dir)
    if ecg_path.exists():
        for f in sorted(ecg_path.glob("*.csv")):
            row, _ = parse_ecg_csv(str(f))
            if row:
                ecg_rows.append(row)
    if ecg_rows:
        ecg_df = pd.DataFrame(ecg_rows)
        io_utils.write_parquet_csv(ecg_df, out / "ecg_summary")
        console.print(f"ECGs: {len(ecg_df)} → {out/'ecg_summary.parquet'}")

    console.rule("QA flags")
    qa = build_qa_flags(workouts, routes_df)
    if not qa.empty:
        io_utils.write_parquet_csv(qa, out / "qa_flags")
        console.print(f"QA flags: {len(qa)} → {out/'qa_flags.parquet'}")

    # Records → daily rollups can be added later (v1 keeps interface ready)
    # records = parse_records(xml_path, types=[
    #     "HeartRate","RestingHeartRate","WalkingHeartRateAverage",
    #     "HeartRateVariabilitySDNN","VO2Max","ActiveEnergyBurned",
    #     "BasalEnergyBurned","AppleExerciseTime","StepCount",
    # ])

    # Heart rate zones & load
    from .parsing.hr import hr_samples_near_workouts
    from .transform.hr_metrics import estimate_hrmax, zones_and_load_for_all

    console.rule("Heart-rate zones & load")
    hr_df = hr_samples_near_workouts(xml_path, workouts)
    if not hr_df.empty:
        if hr_samples:
            io_utils.write_parquet_csv(hr_df, out / "hr_samples")
            console.print(f"HR samples: {len(hr_df)} → {out/'hr_samples.parquet'}")
        hrmax = estimate_hrmax(workouts, hr_df)
        if hrmax:
            zones_df = zones_and_load_for_all(workouts, hr_df, hrmax)
            if not zones_df.empty:
                # merge with stats and overwrite
                stats = stats.merge(zones_df, on="workout_id", how="left")
                io_utils.write_parquet_csv(stats, out / "workout_stats")
                console.print(
                    f"Zones/load added (HRmax≈{round(hrmax)} bpm) → {out/'workout_stats.parquet'}"
                )

    console.rule("Done")
    table = Table(title="Outputs")
    table.add_column("File")
    for f in sorted(out.glob("*.parquet")):
        table.add_row(str(f))
    console.print(table)


@main.command(name="analyze")
@click.option("--xml", "xml_path", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to export.xml")
@click.option("--in-normalized", "in_norm", type=click.Path(exists=True, file_okay=False), default=None, help="Directory with normalized tables")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="out", show_default=True)
@click.option("--tz", "user_tz", default="America/Denver", show_default=True)
@click.option("--weight-goal", type=float, default=None)
@click.option("--step-goal", type=int, default=10000, show_default=True)
@click.option("--stride-m", type=float, default=0.78, show_default=True)
@click.option("--weather", type=click.Choice(["on", "off"]), default="off", show_default=True)
@click.option("--sex", type=click.Choice(["M", "F"]), default="M", show_default=True)
@click.option("--start", type=str, default=None)
@click.option("--end", type=str, default=None)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None, help="YAML config with defaults")
@click.option("--height-cm", type=float, default=None, help="User height in centimeters for BMI if not in data")
def analyze(
    xml_path: str | None,
    in_norm: str | None,
    out_dir: str,
    user_tz: str,
    weight_goal: float | None,
    step_goal: int,
    stride_m: float,
    weather: str,
    sex: str,
    start: str | None,
    end: str | None,
    config_path: str | None,
    height_cm: float | None,
) -> None:
    """Unified pipeline → tables, plots, and Markdown reports."""
    import sys
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from steps.common import StepsConfig as _StepsConfig, normalize_daily as _norm_steps
    from steps.tables import (
        lifetime_summary as _steps_lifetime,
        monthly_rollup as _steps_monthly,
        yearly_rollup as _steps_yearly,
        weekly_rollup as _steps_weekly,
        steps_distribution as _steps_dist,
        dow_stats as _steps_dow,
        top_low_days as _steps_toplow,
        goal_streaks as _steps_streaks,
        steps_correlations as _steps_corr,
    )
    from steps.plots import (
        plot_daily_with_rolling as _plot_steps_daily,
        plot_weekly_steps as _plot_steps_weekly,
        plot_monthly_totals_and_goal_days as _plot_steps_month_goal,
        plot_yearly_totals as _plot_steps_yearly,
        plot_steps_hist as _plot_steps_hist,
        plot_steps_cdf as _plot_steps_cdf,
        plot_dow_average as _plot_steps_dow,
        plot_bucket_distribution as _plot_steps_bucket,
        plot_weekly_goal_pct as _plot_steps_goalpct,
        plot_steps_vs_var as _plot_steps_vs,
    )

    from . import ingest as ingest_mod
    from . import aggregate as agg
    from . import features as feat
    from . import enrich as enr
    from . import plots as P
    from .palette import PALETTE
    from . import reports as R

    out_base = Path(out_dir)
    (out_base / "plots").mkdir(parents=True, exist_ok=True)
    # Ensure category subfolders exist to avoid save errors
    for sub in [
        "summary",
        "running",
        "steps",
        "body",
        "vitals",
        "sleep",
        "alerts",
    ]:
        (out_base / "plots" / sub).mkdir(parents=True, exist_ok=True)
    # Cutoff year for calendar cleanup
    cutoff_year = None
    try:
        cutoff_year = pd.Timestamp(start).year if start else None
    except Exception:
        cutoff_year = None
    (out_base / "tables").mkdir(parents=True, exist_ok=True)
    (out_base / "reports").mkdir(parents=True, exist_ok=True)

    # Load config (optional)
    cfg = {}
    if config_path:
        try:
            import yaml  # type: ignore

            with open(config_path, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            cfg = {}

    cutoff = cfg.get("cutoff_start") if isinstance(cfg, dict) else None
    if not start and cutoff:
        start = str(cutoff)
    if not start:
        start = "2017-01-01"  # hard default per request
    if height_cm is None and isinstance(cfg, dict):
        height_cm = cfg.get("height_cm") or cfg.get("height_cm_default")
    enable_weight_roc = bool(cfg.get("weight_roc")) if isinstance(cfg, dict) else False

    # 1) Ingest
    if in_norm:
        data = ingest_mod.load_normalized(in_norm)
        workouts = data.get("workout_stats", pd.DataFrame())
        routes = data.get("routes", pd.DataFrame())
        hr_samples = data.get("hr_samples", pd.DataFrame())
        daily = data.get("daily_summary", pd.DataFrame())
        ecg = data.get("ecg_summary", pd.DataFrame())
    elif xml_path:
        parsed = ingest_mod.from_export(xml_path, "apple_health_export/workout-routes", "apple_health_export/electrocardiograms", None, user_tz)
        workouts = parsed.get("workout_stats", pd.DataFrame())
        routes = parsed.get("routes", pd.DataFrame())
        hr_samples = pd.DataFrame()  # optional heavy — omit unless precomputed
        daily = parsed.get("daily_summary", pd.DataFrame())
        ecg = parsed.get("ecg_summary", pd.DataFrame())
    else:
        click.echo("[SKIP] No --xml or --in-normalized provided", err=True)
        sys.exit(2)

    # Filter time window if requested
    def _filter_df(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        # Normalize to UTC for robust comparisons
        t = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        if start:
            df = df[t >= pd.Timestamp(start, tz="UTC")].copy()
        if end:
            df = df[t <= (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))].copy()
        return df

    if not workouts.empty:
        ts_col = "start_local" if "start_local" in workouts.columns else "start_utc"
        workouts = _filter_df(workouts, ts_col)
    if not daily.empty and "date" in daily:
        dts = pd.to_datetime(daily["date"], errors="coerce")
        mask = pd.Series(True, index=daily.index)
        if start:
            mask &= dts >= pd.Timestamp(start)
        if end:
            mask &= dts <= (pd.Timestamp(end) + pd.Timedelta(days=1))
        daily = daily[mask].copy()

    # 2) Aggregates (running)
    monthly = agg.running_monthly(workouts)
    weekly = agg.running_weekly(workouts)
    yearly = agg.running_yearly(workouts)
    eff = agg.running_efficiency(workouts)

    # 3) Features: PRs, outliers, zones/TRIMP
    prs, prs_races = feat.compute_prs(workouts)
    out_z, eff_resid, maha = feat.compute_outliers(workouts)
    zt, hrmax, hrrest = feat.compute_zones_trimp(workouts, hr_samples, daily)
    wk_hr = workouts.merge(zt, on="workout_id", how="left") if not zt.empty else workouts.copy()
    wl, ac = feat.compute_load(wk_hr)

    # 4) Weather
    joined_wx = pd.DataFrame(); effects_wx = pd.DataFrame()
    if weather == "on" and not workouts.empty and routes is not None and not routes.empty:
        try:
            joined_wx, effects_wx = enr.join_weather(workouts, routes, cache_dir=str(out_base / "weather_cache"))
        except Exception:
            pass

    # 5) Tables
    from .tables import write_running_tables, write_vitals_tables, write_environment_tables, write_alerts_tables
    write_running_tables(out_base, monthly, weekly, yearly, prs, prs_races, out_z, eff_resid, maha, wl, ac)

    # Vitals daily rollups
    try:
        from analysis.health import daily_rollups, ecg_counts
    except Exception:
        daily_rollups = None  # type: ignore
        ecg_counts = None  # type: ignore

    daily_roll = pd.DataFrame()
    if daily is not None and not daily.empty and daily_rollups is not None:
        try:
            daily_roll = daily_rollups(daily)
        except Exception:
            daily_roll = pd.DataFrame()
    if not daily_roll.empty:
        write_vitals_tables(out_base, daily_roll)

    # Skip environment tables per request
    # if not joined_wx.empty or not effects_wx.empty:
    #     write_environment_tables(out_base, joined_wx, effects_wx)

    # ECG counts table
    if ecg is not None and not ecg.empty and ecg_counts is not None:
        cnt = ecg_counts(ecg)
        if cnt is not None and not cnt.empty:
            write_alerts_tables(out_base, cnt)

    # 6) Plots — Running
    prun = out_base / "plots" / "running"
    # Monthly median pace
    if not monthly.empty:
        # Monthly median pace with improved styling (mm:ss y-axis, faint raw, bold 3-mo median)
        try:
            import matplotlib.pyplot as _plt
            import matplotlib.ticker as _mticker
            x = list(monthly["month"])  # strings like YYYY-MM
            y = pd.to_numeric(monthly["median_pace"], errors="coerce")
            fig, ax = _plt.subplots(figsize=(12, 6))
            # Faint raw
            ax.plot(x, y, color=PALETTE["baseline"], linewidth=1.0, alpha=0.25, marker=None, zorder=1)
            # Bold 3-month rolling median
            roll = y.rolling(3, min_periods=1).median()
            ax.plot(x, roll, color=PALETTE["pace"], linewidth=2.6, zorder=3)
            # Format y as mm:ss
            def _mmss(val: float) -> str:
                if pd.isna(val):
                    return ""
                m = int(val)
                s = int(round((val - m) * 60))
                if s == 60:
                    m += 1; s = 0
                return f"{m:d}:{s:02d}"
            ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda v, pos: _mmss(v)))
            ax.set_title("Monthly Median Pace (lower is faster)")
            ax.set_xlabel("Month")
            ax.set_ylabel("min/mi")
            ax.invert_yaxis()
            # Downsample x ticks for readability
            if len(x) > 18:
                ticks = list(range(0, len(x), max(1, len(x) // 12)))
                ax.set_xticks([x[i] for i in ticks])
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45); lbl.set_ha("right")
            # Clip y-range to 5th–95th percentile with small padding
            try:
                ql, qh = float(y.quantile(0.05)), float(y.quantile(0.95))
                if qh > ql:
                    pad = 0.05 * (qh - ql)
                    ax.set_ylim(qh + pad, ql - pad)  # inverted axis
            except Exception:
                pass
            fig.savefig(prun / "run_monthly_median_pace.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
        except Exception:
            # Fallback to generic line helper
            P.line(
                monthly["month"], monthly["median_pace"],
                title="Monthly Median Pace", xlabel="Month", ylabel="min/mi",
                path=prun / "run_monthly_median_pace.png", color=PALETTE["pace"], invert_y=True, rolling=3, rolling_mode="median", rolling_color=PALETTE["pace"], rotate_x=True, markers=False, y_quantile_clip=(0.05,0.95), raw_alpha=0.2, roll_linewidth=2.6
            )
        # Removed: Monthly Total Miles plot (duplicative with Miles & Runs)
        # Miles and runs: use analysis.viz combined style if available
        try:
            from analysis.viz import line_monthly_miles_and_runs as _mmr
            _mmr(monthly, str(prun / "run_monthly_miles_and_runs.png"))
        except Exception:
            P.bar(monthly["month"], monthly["workouts"], title="Monthly # Runs", xlabel="Month", ylabel="#", path=prun / "run_monthly_miles_and_runs.png", color=PALETTE["counts"])
        # Distance & pace histograms
        try:
            from analysis.viz import hist as _hist
            _hist(workouts["miles"], bins=30, title="Distance Distribution", xlabel="Miles", path=str(prun / "run_hist_distance.png"))
            _hist(workouts["pace_min_per_mile_eff"], bins=30, title="Pace Distribution", xlabel="Pace (min/mi)", path=str(prun / "run_hist_pace.png"))
        except Exception:
            pass
        # Season boxplots and time-of-day effects
        try:
            from analysis.trends import pace_by_season, time_of_day_perf
            from analysis.viz import box_pace_by_season, bar
            seas = pace_by_season(workouts)
            if not seas.empty:
                box_pace_by_season(workouts[["season", "pace_min_per_mile_eff"]].dropna(), prun / "run_pace_by_season_box.png")
                from analysis.viz import bar_miles_by_season
                bar_miles_by_season(seas, prun / "run_miles_by_season.png")
            tod = time_of_day_perf(workouts)
            if not tod.empty:
                bar(tod["tod_bucket"].astype(str).tolist(), tod["median_pace"].tolist(), "Median Pace by Time of Day", "Time of day", "Pace (min/mi)", prun / "run_tod_median_pace_bar.png")
                bar(tod["tod_bucket"].astype(str).tolist(), tod["miles"].tolist(), "Miles by Time of Day", "Time of day", "Miles", prun / "run_tod_miles_bar.png")
        except Exception:
            pass

    if not weekly.empty:
        col = "week_start" if "week_start" in weekly.columns else ("week" if "week" in weekly.columns else weekly.columns[0])
        miles_col = "total_miles" if "total_miles" in weekly.columns else ("miles" if "miles" in weekly.columns else weekly.columns[1])
        P.line(weekly[col], weekly[miles_col], title="Weekly Mileage", xlabel="Week", ylabel="miles", path=prun / "run_weekly_mileage.png", color=PALETTE["volume"], rolling=4, markers=True)

    # Beats-per-mile 30-workout rolling median
    if not workouts.empty and {"pace_min_per_mile_eff"}.issubset(workouts.columns):
        dseq = workouts.sort_values(by=("start_local" if "start_local" in workouts.columns else "start_utc")).copy()
        if "avg_hr_bpm" not in dseq and "avg_hr_bpm_from_samples" in dseq:
            dseq["avg_hr_bpm"] = dseq["avg_hr_bpm_from_samples"]
        if "avg_hr_bpm" in dseq:
            bpmile = pd.to_numeric(dseq["avg_hr_bpm"], errors="coerce") * pd.to_numeric(dseq["pace_min_per_mile_eff"], errors="coerce")
            roll = bpmile.rolling(30, min_periods=5).median()
            P.line(range(len(roll)), roll.values, title="Beats‑per‑Mile (30‑workout rolling)", xlabel="Workout #", ylabel="BPMile", path=prun / "run_beats_per_mile_rolling.png", color=PALETTE["hr"], rolling=None)

    if not workouts.empty and "avg_hr_bpm" in workouts and "pace_min_per_mile_eff" in workouts:
        P.scatter(workouts["avg_hr_bpm"], workouts["pace_min_per_mile_eff"], title="Pace vs HR", xlabel="Avg HR (bpm)", ylabel="Pace (min/mi)", path=prun / "run_pace_vs_hr.png", color=PALETTE["hr"], add_linreg=True)

    # Zones donut
    if not wk_hr.empty and {"z1_min", "z2_min", "z3_min", "z4_min", "z5_min"}.issubset(wk_hr.columns):
        totals = wk_hr[["z1_min", "z2_min", "z3_min", "z4_min", "z5_min"]].sum()
        P.donut(["Z1", "Z2", "Z3", "Z4", "Z5"], totals.tolist(), title="Zone Distribution", path=prun / "run_zone_distribution.png")

    if not wl.empty:
        xcol = "week" if "week" in wl.columns else ("week_start" if "week_start" in wl.columns else wl.columns[0])
        ycol = "weekly_trimp" if "weekly_trimp" in wl.columns else ("trimp" if "trimp" in wl.columns else wl.columns[1])
        P.line(wl[xcol], wl[ycol], title="Weekly TRIMP", xlabel="Week", ylabel="TRIMP", path=prun / "run_weekly_trimp.png", color=PALETTE["hr"], rolling=4, markers=True)
    if not ac.empty and "ac_ratio" in ac:
        xcol = "week" if "week" in ac.columns else ("week_start" if "week_start" in ac.columns else ac.columns[0])
        P.line(ac[xcol], ac["ac_ratio"], title="Acute:Chronic Ratio", xlabel="Week", ylabel="A:C", path=prun / "run_acute_chronic.png", color=PALETTE["hr"], rolling=None, markers=True)

    # HR drift if samples exist
    try:
        from analysis.outliers import hr_drift as _hr_drift
        if hr_samples is not None and not hr_samples.empty:
            dr = _hr_drift(hr_samples, workouts)
            if not dr.empty and "hr_drift_pct" in dr:
                P.scatter(dr["distance_m"].astype(float) / 1609.34, dr["hr_drift_pct"], title="HR Drift vs Distance", xlabel="Distance (mi)", ylabel="Drift %", path=prun / "run_hr_drift_vs_distance.png", color=PALETTE["hr"], add_linreg=True)
                # Histogram
                import matplotlib.pyplot as _plt
                fig, ax = _plt.subplots(figsize=(12, 7)); ax.hist(pd.to_numeric(dr["hr_drift_pct"], errors="coerce").dropna().values, bins=30, color=PALETTE["hr"], edgecolor=None); ax.set_title("HR Drift % Histogram"); ax.set_xlabel("Drift %"); ax.set_ylabel("Count"); fig.savefig(prun / "run_hr_drift_hist.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
    except Exception:
        pass

    # Elevation scatter if elev available
    if not workouts.empty and {"elev_gain_m", "miles"}.issubset(workouts.columns):
        elev_per_mile = (pd.to_numeric(workouts["elev_gain_m"], errors="coerce") / workouts["miles"].replace(0, pd.NA)).astype(float)
        P.scatter(elev_per_mile, workouts["pace_min_per_mile_eff"], title="Pace vs Elevation Gain per Mile", xlabel="Elev/mile (m)", ylabel="Pace (min/mi)", path=prun / "run_pace_vs_elev_per_mile.png", color=PALETTE["elevation"], add_linreg=True)

    # Route clusters & starts
    try:
        if routes is not None and not routes.empty:
            from analysis.routes import route_start_clusters, route_cluster_perf
            from analysis.viz import bar_route_clusters, scatter_points
            clusters = route_start_clusters(routes)
            perf = route_cluster_perf(workouts, routes)
            if not perf.empty:
                bar_route_clusters(perf, prun / "run_route_clusters.png")
            # start locations scatter
            if {"start_lat", "start_lon"}.issubset(routes.columns):
                xs = pd.to_numeric(routes["start_lon"], errors="coerce").dropna()
                ys = pd.to_numeric(routes["start_lat"], errors="coerce").dropna()
                if not xs.empty and not ys.empty:
                    # simple scatter fallback
                    import matplotlib.pyplot as _plt
                    fig, ax = _plt.subplots(figsize=(12, 7)); ax.scatter(xs, ys, s=8, alpha=0.5, color=PALETTE["volume"]); ax.set_title("Start Locations"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); fig.savefig(prun / "run_start_locations_scatter.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
    except Exception:
        pass

    # Device bias + correlation matrix reuse existing outputs when present
    try:
        from analysis.devices import device_bias, device_bucket_matrix
        db = device_bias(workouts)
        if not db.empty:
            from analysis.viz import bar_device_bias, heatmap_matrix
            bar_device_bias(db, out_base / "plots" / "running" / "run_device_bias_bar.png")
            mat = device_bucket_matrix(workouts)
            if mat is not None and not mat.empty:
                heatmap_matrix(mat, "Device × Bucket Pace Delta", "Bucket", "Device", out_base / "plots" / "running" / "run_device_bucket_heatmap.png")
    except Exception:
        pass

    # Correlation matrix (Spearman via rank-Pearson)
    num_cols = [
        c
        for c in [
            "miles",
            "pace_min_per_mile_eff",
            "avg_hr_bpm",
            "max_hr_bpm",
            "moving_time_sec_eff",
            "z1_min",
            "z2_min",
            "z3_min",
            "z4_min",
            "z5_min",
            "trimp",
        ]
        if c in workouts
    ]
    if len(num_cols) >= 2:
        corrm = workouts[num_cols].rank(pct=True).corr(method="pearson")
        corrm.to_csv(out_base / "tables" / "running" / "run_correlation_matrix.csv")
        try:
            from analysis.viz import heatmap_matrix
            heatmap_matrix(corrm, "Correlation Matrix", "Features", "Features", out_base / "plots" / "running" / "run_correlation_matrix.png")
        except Exception:
            # Fallback simple heatmap using our helper
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(12, 7))
            im = ax.imshow(corrm.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corrm.columns))); ax.set_xticklabels(corrm.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corrm.index))); ax.set_yticklabels(corrm.index)
            fig.colorbar(im, ax=ax, label="ρ"); ax.set_title("Correlation Matrix")
            fig.savefig(out_base / "plots" / "running" / "run_correlation_matrix.png", dpi=144, bbox_inches="tight"); _plt.close(fig)

    # PR timeline
    if prs is not None and not prs.empty:
        try:
            from analysis.viz import timeline_prs
            timeline_prs(prs, out_base / "plots" / "running" / "run_pr_timeline.png")
        except Exception:
            pass

    # 7) Steps — tables + plots
    if daily is not None and not daily.empty and "steps" in daily.columns:
        scfg = _StepsConfig(step_goal=step_goal, stride_m=stride_m)
        dsteps = _norm_steps(daily, scfg)
        s_tables: dict[str, pd.DataFrame] = {
            "steps_lifetime_summary": _steps_lifetime(dsteps, scfg),
            "steps_monthly": _steps_monthly(dsteps),
            "steps_yearly": _steps_yearly(dsteps),
            "steps_weekly": _steps_weekly(dsteps),
            "steps_distribution": _steps_dist(dsteps),
            "steps_dayofweek": _steps_dow(dsteps),
            "steps_streaks": _steps_streaks(dsteps, scfg),
        }
        top, low = _steps_toplow(dsteps)
        s_tables["steps_top_days"] = top
        s_tables["steps_low_days"] = low
        corr_df = _steps_corr(dsteps)
        if corr_df is not None and not corr_df.empty:
            s_tables["steps_correlations"] = corr_df
        from .tables import write_steps_tables
        write_steps_tables(out_base, s_tables)
        pdir = out_base / "plots" / "steps"
        _plot_steps_daily(dsteps, pdir)
        _plot_steps_weekly(_steps_weekly(dsteps), pdir)
        _plot_steps_month_goal(_steps_monthly(dsteps), scfg, pdir)
        _plot_steps_yearly(_steps_yearly(dsteps), pdir)
        _plot_steps_hist(dsteps, pdir)
        _plot_steps_cdf(dsteps, pdir)
        _plot_steps_dow(_steps_dow(dsteps), pdir)
        _plot_steps_bucket(_steps_dist(dsteps), pdir)
        _plot_steps_goalpct(_steps_weekly(dsteps), pdir)
        for v in ["active_kcal", "exercise_min", "rhr_bpm", "vo2max_mlkgmin", "walking_hr_avg_bpm"]:
            _plot_steps_vs(dsteps, v, pdir)
        # Steps calendar + cumulative
        try:
            # Cleanup old calendars before regenerating
            if cutoff_year is not None:
                for p in (pdir).glob("calendar_steps_*.png"):
                    try:
                        yr = int(p.stem.split("_")[-1])
                        if yr < cutoff_year:
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
            years = pd.to_datetime(dsteps["date"]).dt.year.dropna().astype(int).unique().tolist()
            vmax = float(pd.to_numeric(dsteps["steps"], errors="coerce").quantile(0.95)) if not dsteps.empty else None
            for y in years:
                from steps.plots import plot_calendar_heatmap, plot_cumulative_vs_goal
                plot_calendar_heatmap(dsteps, y, pdir, vmin=0.0, vmax=vmax)
                plot_cumulative_vs_goal(dsteps, y, scfg, pdir)
        except Exception:
            pass

    # 7b) Workouts calendar (miles per day) — use Steps renderer with proper labels
    try:
        if not workouts.empty:
            from analysis.common import add_effective_fields
            from steps.plots import plot_calendar_heatmap as cal_heatmap
            w = add_effective_fields(workouts.copy())
            dt_all = pd.to_datetime(w.get("start_local", w.get("start_utc")), utc=True, errors="coerce")
            miles = pd.to_numeric(w.get("miles"), errors="coerce")
            df = pd.DataFrame({"date": dt_all.dt.tz_convert(None).dt.normalize(), "steps": miles}).dropna()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            daily_miles = df.groupby("date")["steps"].sum()
            vmax = float(pd.Series(daily_miles.values).quantile(0.95)) if not daily_miles.empty else None
            # Cleanup old misnamed running calendars
            for p in prun.glob("calendar_steps_*.png"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            years = sorted(pd.to_datetime(df["date"]).dt.year.unique().astype(int).tolist())
            for yr in years:
                cal_heatmap(
                    df.rename(columns={"steps": "steps"}),
                    yr,
                    prun,
                    vmin=0.0,
                    vmax=vmax,
                    title_prefix="Runs",
                    metric_label="Miles",
                    filename_prefix="calendar_runs",
                )
    except Exception as e:
        click.echo(f"[WARN] running calendars: {e}", err=True)

    # 8) Vitals plots
    pvit = out_base / "plots" / "vitals"
    if not daily_roll.empty:
        if {"date", "rhr_bpm"}.issubset(daily_roll.columns):
            P.line(
                daily_roll["date"],
                pd.to_numeric(daily_roll["rhr_bpm"], errors="coerce"),
                title="Resting HR (7D)",
                xlabel="Date",
                ylabel="bpm",
                path=pvit / "vitals_rhr_trend.png",
                color=PALETTE["hr"],
                rolling=7,
                rolling_mode="mean",
                y_quantile_clip=(0.05, 0.95),
                raw_alpha=0.2,
            )
        if {"date", "hrv_sdnn_ms"}.issubset(daily_roll.columns):
            P.line(
                daily_roll["date"],
                pd.to_numeric(daily_roll["hrv_sdnn_ms"], errors="coerce"),
                title="HRV SDNN (7D)",
                xlabel="Date",
                ylabel="ms",
                path=pvit / "vitals_hrv_trend.png",
                color=PALETTE["hrv"],
                rolling=7,
                rolling_mode="mean",
                y_quantile_clip=(0.05, 0.95),
                raw_alpha=0.2,
            )
        if {"date", "vo2max_mlkgmin"}.issubset(daily_roll.columns):
            # Resample & interpolate raw → daily, then draw faint raw + bold 30D mean
            vo = pd.to_numeric(daily_roll["vo2max_mlkgmin"], errors="coerce")
            ser = pd.Series(vo.values, index=pd.to_datetime(daily_roll["date"]))
            ser = ser.sort_index().resample("D").mean().interpolate("time")
            P.line(
                ser.index,
                ser.values,
                title="VO2max (30D)",
                xlabel="Date",
                ylabel="mL/kg·min",
                path=pvit / "vitals_vo2_trend.png",
                color=PALETTE["vo2"],
                rolling=30,
                rolling_mode="mean",
                markers=False,
                y_quantile_clip=(0.05, 0.95),
                raw_alpha=0.2,
            )
        # Additional vitals
        if "walking_hr_avg_bpm" in daily_roll:
            wh = pd.to_numeric(daily_roll["walking_hr_avg_bpm"], errors="coerce")
            ser = pd.Series(wh.values, index=pd.to_datetime(daily_roll["date"]))
            ser = ser.sort_index().resample("D").mean().interpolate("time")
            P.line(
                ser.index,
                ser.values,
                title="Walking HR Avg (7D)",
                xlabel="Date",
                ylabel="bpm",
                path=pvit / "vitals_walking_hr_avg_trend.png",
                color=PALETTE["hr"],
                rolling=7,
                rolling_mode="mean",
                markers=False,
                y_quantile_clip=(0.05, 0.95),
                raw_alpha=0.2,
            )
        if {"bp_systolic", "bp_diastolic"}.issubset(daily_roll.columns):
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(12, 7));
            x = pd.to_datetime(daily_roll["date"]) ;
            ax.plot(x, pd.to_numeric(daily_roll["bp_systolic"], errors="coerce"), color=PALETTE["hr"], label="Systolic")
            ax.plot(x, pd.to_numeric(daily_roll["bp_diastolic"], errors="coerce"), color=PALETTE["baseline"], label="Diastolic")
            ax.set_title("Blood Pressure"); ax.set_xlabel("Date"); ax.set_ylabel("mmHg"); ax.legend(); fig.savefig(pvit / "vitals_bp_trend.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
        if "respiratory_rate_bpm" in daily_roll:
            P.line(
                daily_roll["date"],
                pd.to_numeric(daily_roll["respiratory_rate_bpm"], errors="coerce"),
                title="Respiratory Rate (7D)",
                xlabel="Date",
                ylabel="bpm",
                path=pvit / "vitals_resp_rate_trend.png",
                color=PALETTE["baseline"],
                rolling=7,
                markers=False,
                y_quantile_clip=(0.05, 0.95),
                raw_alpha=0.2,
            )
        if "spo2_pct" in daily_roll:
            import matplotlib.pyplot as _plt
            s = pd.to_numeric(daily_roll["spo2_pct"], errors="coerce").dropna()
            if not s.empty:
                fig, ax = _plt.subplots(figsize=(12,7)); ax.hist(s, bins=30, color=PALETTE["baseline"], edgecolor=None); ax.set_title("SpO₂ Distribution"); ax.set_xlabel("SpO₂ %"); ax.set_ylabel("Count"); fig.savefig(pvit / "vitals_spo2_hist.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
        # Cross metrics
        if {"rhr_bpm", "vo2max_mlkgmin"}.issubset(daily_roll.columns):
            P.scatter(pd.to_numeric(daily_roll["rhr_bpm"], errors="coerce"), pd.to_numeric(daily_roll["vo2max_mlkgmin"], errors="coerce"), title="RHR vs VO₂max", xlabel="RHR (bpm)", ylabel="VO₂ (mL/kg·min)", path=pvit / "vitals_rhr_vs_vo2.png", color=PALETTE["vo2"], add_linreg=True)
        if not wl.empty and "hrv_sdnn_ms" in daily_roll:
            # Map weekly TRIMP to dates by week start for scatter (rough join)
            try:
                wl2 = wl.copy(); wl2["week"] = pd.to_datetime(wl2.get("week", wl2.columns[0])) ;
                dr = daily_roll.copy(); dr["week"] = pd.to_datetime(dr["date"]).dt.to_period("W-MON").dt.start_time
                merged = dr.merge(wl2[["week", wl2.columns[-1]]], on="week", how="left")
                P.scatter(pd.to_numeric(merged.iloc[:, -1], errors="coerce"), pd.to_numeric(merged["hrv_sdnn_ms"], errors="coerce"), title="HRV vs Weekly TRIMP", xlabel="Weekly TRIMP", ylabel="HRV (ms)", path=pvit / "vitals_hrv_vs_trimp.png", color=PALETTE["hrv"], add_linreg=True)
            except Exception:
                pass

    # 9) Body (weight/bodyfat/BMI) if present
    pbody = out_base / "plots" / "body"
    tbody = out_base / "tables" / "body"
    tbody.mkdir(parents=True, exist_ok=True)
    if daily is not None and not daily.empty:
        # Ensure datetime x-axis for consistent rendering
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        debug = pd.DataFrame({"date": daily["date"]})
        if "weight_kg" in daily.columns:
            w = pd.to_numeric(daily["weight_kg"], errors="coerce")
            w = w.where(w > 0)  # treat 0/neg as missing
            # Heuristic: convert if values appear in pounds
            try:
                med = float(w.dropna().median()) if w.notna().any() else None
                if med and med > 150:  # likely pounds
                    w = w * 0.45359237
            except Exception:
                pass
            debug["weight_plot_kg"] = w
            # Interpolate daily to connect bridges
            w_ser = pd.Series(w.values, index=daily["date"]).sort_index()
            w_daily = w_ser.resample("D").mean().interpolate(method="time", limit_area="inside")
            w_lb = w_daily * 2.20462
            P.line(
                w_lb.index, w_lb.values,
                title="Weight (lb)", xlabel="Date", ylabel="lb",
                path=pbody / "body_weight_trend.png", color=PALETTE["counts"],
                rolling=30, rolling_mode="mean", rotate_x=True, markers=False, y_quantile_clip=(0.05,0.95), grid=False
            )
            # Rate of change (30d slope)
            if enable_weight_roc:
                try:
                    import numpy as _np
                    d = daily.copy(); d = d.sort_values("date")
                    wkg = pd.to_numeric(d["weight_kg"], errors="coerce").where(lambda s: s>0)
                    # convert lb→kg if needed
                    med = float(wkg.dropna().median()) if wkg.notna().any() else None
                    if med and med > 150:
                        wkg = wkg * 0.45359237
                    if wkg.notna().sum() >= 10:
                        win_days = 30
                        slopes=[]
                        for i in range(len(d)):
                            t_end = d["date"].iloc[i]
                            t_start = t_end - pd.Timedelta(days=win_days)
                            mask = (d["date"]>=t_start) & (d["date"]<=t_end)
                            ww = wkg[mask].dropna()
                            tt = d.loc[mask, "date"][ww.index]
                            if len(ww) >= 5:
                                xdays = (tt - tt.min()).dt.total_seconds()/86400.0
                                m, b = _np.polyfit(xdays, ww.values, 1)  # kg/day
                                slopes.append(m * 7 * 2.20462)  # lb/week
                            else:
                                slopes.append(_np.nan)
                        P.line(d["date"], slopes, title="Weight Rate of Change (lb/week)", xlabel="Date", ylabel="lb/week", path=pbody / "body_weight_rate_of_change.png", color=PALETTE["counts"], rolling=7, rolling_mode="mean", markers=False)
                except Exception:
                    pass
        if "body_fat_pct" in daily.columns:
            bf = pd.to_numeric(daily["body_fat_pct"], errors="coerce").where(lambda s: (s > 0) & (s < 80))
            # interpolate daily to connect bridges
            bf_ser = pd.Series(bf.values, index=daily["date"]).sort_index()
            bf_daily = bf_ser.resample("D").mean().interpolate(method="time", limit_area="inside")
            bf_pct = bf_daily * 100.0
            debug["bodyfat_plot_pct"] = bf_pct
            P.line(bf_pct.index, bf_pct.values, title="Body Fat %", xlabel="Date", ylabel="%", path=pbody / "body_fat_trend.png", color=PALETTE["counts"], rolling=30, rolling_mode="mean", rotate_x=True, markers=False, y_quantile_clip=(0.05,0.95), grid=False)
        if {"weight_kg", "body_fat_pct"}.issubset(daily.columns):
            P.scatter(pd.to_numeric(daily["weight_kg"], errors="coerce"), pd.to_numeric(daily["body_fat_pct"], errors="coerce"), title="Body Fat % vs Weight", xlabel="Weight (kg)", ylabel="Body Fat %", path=pbody / "body_bodyfat_vs_weight.png", color=PALETTE["counts"], add_linreg=True)
        if "height_cm" in daily.columns and "weight_kg" in daily.columns:
            import numpy as _np
            # Infer height units → meters
            h_raw = pd.to_numeric(daily["height_cm"], errors="coerce")
            h_candidates = h_raw.dropna()
            h_m = None
            if not h_candidates.empty:
                medh = float(h_candidates.median())
                s = pd.Series(h_candidates)
                if 10 < medh < 300:  # likely centimeters
                    h_m = s / 100.0
                elif 3.0 < medh < 8.0:  # likely feet
                    h_m = s * 0.3048
                elif 30 < medh < 120:  # likely inches
                    h_m = s * 0.0254
                elif 0.5 < medh < 3.0:  # already meters
                    h_m = s
            if h_m is None:
                h_m = pd.Series(index=daily.index, dtype=float)
            h_m = h_m.where(h_m > 0)
            h_fixed = float(h_m.dropna().median()) if h_m.notna().any() else None
            w = pd.to_numeric(daily["weight_kg"], errors="coerce").where(lambda s: s > 0)
            try:
                med = float(w.dropna().median()) if w.notna().any() else None
                if med and med > 150:
                    w = w * 0.45359237
            except Exception:
                pass
            if not h_fixed and height_cm:
                h_fixed = float(height_cm) / 100.0
            # compute BMI at measurement dates
            if h_fixed:
                bmi_pts = (w / (h_fixed * h_fixed)).replace([_np.inf, -_np.inf], _np.nan)
            else:
                bmi_pts = (w / (h_m.reindex(daily.index) * h_m.reindex(daily.index))).replace([_np.inf, -_np.inf], _np.nan)
            # interpolate daily to connect bridges
            bmi_ser = pd.Series(bmi_pts.values, index=daily["date"]).sort_index()
            bmi_daily = bmi_ser.resample("D").mean().interpolate(method="time", limit_area="inside")
            debug["bmi_plot"] = bmi_daily
            if (bmi_daily.notna().sum() or 0) > 0:
                P.line(bmi_daily.index, bmi_daily.values, title="BMI", xlabel="Date", ylabel="kg/m²", path=pbody / "body_bmi_trend.png", color=PALETTE["counts"], rolling=30, rolling_mode="mean", rotate_x=True, markers=False, y_quantile_clip=(0.05,0.95), grid=False)
            else:
                click.echo("[SKIP] BMI plot: no height available; set --height-cm or config height_cm", err=True)
        # write debug
        try:
            debug.to_csv(tbody / "body_plot_data.csv", index=False)
        except Exception:
            pass
        # Monthly weight summary plot removed per request

    # 10) Sleep
    psleep = out_base / "plots" / "sleep"
    if daily is not None and not daily.empty and {"sleep_asleep_min", "sleep_in_bed_min"}.issubset(daily.columns):
        import matplotlib.pyplot as _plt
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        (out_base / "plots" / "sleep").mkdir(parents=True, exist_ok=True)
        fig, ax = _plt.subplots(figsize=(12, 7))
        ax.bar(d["date"], d["sleep_in_bed_min"], color=PALETTE["baseline"], label="In bed")
        ax.bar(d["date"], d["sleep_asleep_min"], color=PALETTE["volume"], label="Asleep")
        ax.set_title("Sleep Duration (Asleep vs In bed)"); ax.set_xlabel("Date"); ax.set_ylabel("Minutes"); ax.legend(); fig.savefig(psleep / "sleep_duration_stacked.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
        # Efficiency
        eff_sleep = (pd.to_numeric(d["sleep_asleep_min"], errors="coerce") / pd.to_numeric(d["sleep_in_bed_min"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
        P.line(d["date"], eff_sleep, title="Sleep Efficiency", xlabel="Date", ylabel="efficiency (0–1)", path=psleep / "sleep_efficiency_trend.png", color=PALETTE["counts"], rolling=7, markers=True, y_quantile_clip=(0.05,0.95))
        # Sleep calendars + DoW avg + correlations
        try:
            years = d["date"].dt.year.dropna().astype(int).unique().tolist()
            # Calendar heatmap for sleep (minutes asleep)
            import numpy as _np
            import matplotlib.pyplot as _plt
            # Cleanup old calendars
            if cutoff_year is not None:
                for p in (psleep).glob("sleep_calendar_*.png"):
                    try:
                        yr = int(p.stem.split("_")[-1])
                        if yr < cutoff_year:
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
            for y in years:
                dd = d[d["date"].dt.year == y]
                vals = dd.set_index("date")["sleep_asleep_min"]
                start = pd.Timestamp(f"{y}-01-01")
                idx = pd.date_range(start, start + pd.offsets.YearEnd(), freq="D")
                vals = vals.reindex(idx)
                weeks = (len(idx) + start.weekday()) // 7 + 1
                mat = _np.zeros((7, weeks))
                for i, dt in enumerate(idx):
                    w = (start.weekday() + i) // 7
                    d0 = (start.weekday() + i) % 7
                    mat[d0, w] = float(vals.get(dt, _np.nan) or 0.0)
                fig, ax = _plt.subplots(figsize=(12, 3)); im=ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest"); ax.set_title(f"Sleep Calendar {y}"); fig.colorbar(im, ax=ax, label="Minutes asleep"); ax.axis("off"); fig.savefig(psleep / f"sleep_calendar_{y}.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
            # DoW avg
            dow = d.copy(); dow["dow"] = dow["date"].dt.day_name(); g = dow.groupby("dow")["sleep_asleep_min"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(10,4)); ax.bar(g.index, g.values, color=PALETTE["volume"]); ax.set_title("Avg Sleep by Weekday"); ax.set_xlabel(""); ax.set_ylabel("Minutes asleep"); fig.savefig(psleep / "sleep_dow_avg.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
            # Sleep vs next-day steps & RHR
            if {"steps","rhr_bpm"}.issubset(daily.columns):
                d2 = daily.copy(); d2["date"] = pd.to_datetime(d2["date"]) ; d2 = d2.sort_values("date")
                nxt = d2[["date","steps","rhr_bpm"]].copy(); nxt["date"] = nxt["date"] - pd.Timedelta(days=1)
                merged = d2.merge(nxt, on="date", suffixes=("", "_next"))
                P.scatter(pd.to_numeric(merged["sleep_asleep_min"], errors="coerce"), pd.to_numeric(merged["steps_next"], errors="coerce"), title="Sleep vs Next-day Steps", xlabel="Sleep minutes", ylabel="Next-day steps", path=psleep / "sleep_vs_steps.png", color=PALETTE["volume"], add_linreg=True)
                P.scatter(pd.to_numeric(merged["sleep_asleep_min"], errors="coerce"), pd.to_numeric(merged["rhr_bpm_next"], errors="coerce"), title="Sleep vs Next-day RHR", xlabel="Sleep minutes", ylabel="Next-day RHR (bpm)", path=psleep / "sleep_vs_rhr.png", color=PALETTE["hr"], add_linreg=True)
            # Sleep debt rolling vs 7h
            baseline_min = 7*60
            debt = pd.to_numeric(d["sleep_asleep_min"], errors="coerce") - baseline_min
            P.line(d["date"], debt.rolling(7, min_periods=3).mean(), title="Sleep Debt (rolling, vs 7h)", xlabel="Date", ylabel="Minutes", path=psleep / "sleep_debt_rolling.png", color=PALETTE["baseline"], rolling=None, markers=True, y_quantile_clip=(0.05,0.95))
        except Exception:
            pass

    # 11) Alerts
    palerts = out_base / "plots" / "alerts"
    if ecg is not None and not ecg.empty:
        try:
            from analysis.health import ecg_counts
            cnt = ecg_counts(ecg)
            if not cnt.empty:
                from analysis.viz import bar_counts
                bar_counts(cnt, "classification", "count", "ECG Classification Counts", palerts / "ecg_counts.png")
            # ECG HR histogram if mean_hr_bpm present
            if "mean_hr_bpm" in ecg.columns:
                import matplotlib.pyplot as _plt
                s = pd.to_numeric(ecg["mean_hr_bpm"], errors="coerce").dropna()
                if not s.empty:
                    fig, ax = _plt.subplots(figsize=(10,4)); ax.hist(s, bins=30, color=PALETTE["hr"], edgecolor=None); ax.set_title("ECG Mean HR Distribution"); ax.set_xlabel("bpm"); ax.set_ylabel("Count"); fig.savefig(palerts / "ecg_hr_hist.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
        except Exception:
            pass

    # 12) Environment plots — removed per request

    # 13) Summary KPI (simple text tiles)
    try:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(12, 7)); ax.axis("off")
        year = int(pd.to_datetime(workouts.get("start_local", workouts.get("start_utc")), utc=True).dt.year.max()) if not workouts.empty else datetime.now().year
        total_miles = float(monthly["total_miles"].sum()) if not monthly.empty else 0.0
        runs = int(monthly["workouts"].sum()) if not monthly.empty else 0
        pace = float(monthly["median_pace"].median()) if not monthly.empty else float("nan")
        steps_goal = None
        if daily is not None and not daily.empty and "steps" in daily.columns:
            dsteps = _norm_steps(daily, _StepsConfig(step_goal=step_goal, stride_m=stride_m))
            steps_goal = 100.0 * dsteps.get("goal_hit", pd.Series(dtype=float)).mean()
        lines = [
            f"Year: {year}",
            f"Miles: {total_miles:.1f}",
            f"Runs: {runs}",
            f"Median Pace: {int(pace) if pd.notna(pace) else '—'}:{int(round((pace - int(pace))*60)) if pd.notna(pace) else '--':02d} min/mi",
            f"Steps Goal Hit: {steps_goal:.1f}%" if steps_goal is not None else "Steps Goal Hit: —",
        ]
        ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=14)
        fig.savefig(out_base / "plots" / "summary" / "summary_kpis.png", dpi=144, bbox_inches="tight"); _plt.close(fig)
    except Exception:
        pass

    # 13b) Summary charts: workouts by activity and top sources
    try:
        psum = out_base / "plots" / "summary"
        psum.mkdir(parents=True, exist_ok=True)
        if not workouts.empty:
            # Activity counts
            act_counts = (
                workouts.get("activity").fillna("Unknown").astype(str).value_counts()
            )
            from .plots import bar as _bar
            _bar(
                list(act_counts.index),
                list(act_counts.values),
                title="Workouts by Activity",
                xlabel="Activity",
                ylabel="#",
                path=psum / "summary_workouts_by_activity.png",
                color=PALETTE["counts"],
            )
            # Top sources (devices/apps)
            src_counts = (
                workouts.get("source_name").fillna("Unknown").astype(str).value_counts().head(10)
            )
            _bar(
                list(src_counts.index),
                list(src_counts.values),
                title="Top Sources (# Workouts)",
                xlabel="Source",
                ylabel="#",
                path=psum / "summary_sources_top10.png",
                color=PALETTE["baseline"],
            )
    except Exception:
        pass

    # 14) Reports
    R.write_master_index(out_base)
    R.write_running(out_base)
    R.write_steps(out_base)
    R.write_body(out_base)
    R.write_vitals(out_base)
    R.write_sleep(out_base)
    R.write_alerts(out_base)
    # Environment section removed


# Wire in extended health analytics (full metrics suite)
try:
    from health_analysis.cli import analyze_health as analyze_health_cmd  # type: ignore

    main.add_command(analyze_health_cmd)
except Exception:
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
