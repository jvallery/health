from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import numpy as np

from . import viz
from .common import add_effective_fields, normalize_workouts
from .devices import device_bias
from .load import acute_chronic_ratio, weekly_load
from .outliers import efficiency_residuals, multivariate_outliers, pace_zscores_by_bucket
from .prs import pr_by_distance_bucket
from .reports import (
    annual_report,
    monthly_report,
    write_markdown_suite,
    write_plots_catalog,
    write_summary_md,
)
from .routes import route_cluster_perf, route_start_clusters
from .trends import (
    efficiency_trend,
    monthly_perf,
    pace_by_season,
    streaks_summary,
    time_of_day_perf,
    weekly_volume,
    yearly_perf,
)
from .weather import annotate_weather, weather_effects
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


def _read_df(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(stem)


def run_all(
    indir: str, outdir: str, tz: str = "America/Denver", with_weather: bool = False
) -> None:
    in_path = Path(indir)
    out_path = Path(outdir) / "analysis"
    out_csv = out_path / "tables"
    out_plots = out_path / "plots"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    workouts = _read_df(in_path, "workout_stats")
    # Try to enrich avg_hr_bpm from hr_samples for efficiency plots
    try:
        _hrs = _read_df(in_path, "hr_samples")
    except FileNotFoundError:
        _hrs = pd.DataFrame()
    if not _hrs.empty and "workout_id" in _hrs:
        avg_hr = (
            _hrs.groupby("workout_id")["bpm"].mean().rename("avg_hr_bpm_from_samples")
        )
        workouts = workouts.merge(avg_hr, on="workout_id", how="left")
        # Fill avg_hr_bpm when missing
        if "avg_hr_bpm" in workouts:
            workouts["avg_hr_bpm"] = workouts["avg_hr_bpm"].fillna(
                workouts["avg_hr_bpm_from_samples"]
            )
        else:
            workouts["avg_hr_bpm"] = workouts["avg_hr_bpm_from_samples"]
    workouts = normalize_workouts(workouts, tz)
    routes = None
    try:
        routes = _read_df(in_path, "routes")
    except FileNotFoundError:
        routes = None

    # Trends
    m = monthly_perf(workouts)
    y = yearly_perf(workouts)
    e = efficiency_trend(workouts)
    s = pace_by_season(workouts)
    t = time_of_day_perf(workouts)
    w = weekly_volume(workouts)
    streaks = streaks_summary(workouts)

    m.to_csv(out_csv / "monthly_perf.csv", index=False)
    y.to_csv(out_csv / "yearly_perf.csv", index=False)
    e.to_csv(out_csv / "efficiency_trend.csv", index=False)
    s.to_csv(out_csv / "pace_by_season.csv", index=False)
    t.to_csv(out_csv / "time_of_day_perf.csv", index=False)
    w.to_csv(out_csv / "weekly_volume.csv", index=False)
    (out_csv / "streaks_summary.json").write_text(pd.Series(streaks).to_json())

    # Plots
    viz.line_monthly_pace(m, out_plots / "monthly_pace.png")
    viz.line_monthly_miles(m, out_plots / "monthly_miles.png")
    viz.line_efficiency(e, out_plots / "efficiency_trend.png")
    # For season box, we need row-level pace values with season label
    ws = workouts.copy()
    dt_local = pd.to_datetime(ws.get("start_local", ws.get("start_utc")), utc=True)
    ws["season"] = dt_local.dt.month.map(
        lambda m: (
            "DJF"
            if m in (12, 1, 2)
            else ("MAM" if m in (3, 4, 5) else ("JJA" if m in (6, 7, 8) else "SON"))
        )
    )
    viz.box_pace_by_season(
        ws[["season", "pace_min_per_mile_eff"]].dropna(), out_plots / "pace_by_season.png"
    )
    viz.bar_miles_by_season(s, out_plots / "miles_by_season.png")
    # Monthly combined
    mm = monthly_perf(workouts)
    out_mm_path = out_plots / "monthly_miles_and_runs.png"
    viz.line_monthly_miles_and_runs(mm, out_mm_path)
    # Save simple table for dashboarding
    mm[["month", "total_miles", "workouts"]].to_csv(
        out_csv / "monthly_miles_and_runs.csv", index=False
    )

    # Variant: kilometers
    mm_km = mm.copy()
    mm_km["total_km"] = mm_km["total_miles"] * 1.60934
    viz.line_monthly_km_and_runs(mm_km, out_plots / "monthly_km_and_runs.png")

    # Variant: last 24 months
    mm2 = mm.copy()
    mm2["dt"] = pd.to_datetime(mm2["month"] + "-01")
    mm_last24 = mm2.sort_values("dt").tail(24).drop(columns=["dt"])
    viz.line_monthly_miles_and_runs(mm_last24, out_plots / "monthly_miles_and_runs_last24.png")
    mm_last24[["month", "total_miles", "workouts"]].to_csv(
        out_csv / "monthly_miles_and_runs_last24.csv", index=False
    )

    # Unified dataset: remove per-activity charts (run/walk/treadmill merged)

    # Time-of-day heatmap
    dt = pd.to_datetime(workouts["start_dt_local"], utc=True)
    counts = pd.crosstab(dt.dt.dayofweek, dt.dt.hour, dropna=False).reindex(
        index=range(7), columns=range(24), fill_value=0
    )
    viz.heatmap_dow_hour(counts, out_plots / "dow_hour_heatmap.png")

    # Histograms
    viz.hist(
        workouts["miles"],
        bins=30,
        title="Distance Distribution",
        xlabel="Miles",
        path=str(out_plots / "hist_distance.png"),
    )
    viz.hist(
        workouts["pace_min_per_mile_eff"],
        bins=30,
        title="Pace Distribution",
        xlabel="Pace (min/mi)",
        path=str(out_plots / "hist_pace.png"),
    )

    # Routes & devices & PRs
    if routes is not None and not routes.empty:
        clusters = route_start_clusters(routes)
        perf = route_cluster_perf(workouts, routes)
        # Optional: reverse-geocode cluster labels with lightweight caching
        try:
            cache_path = out_csv / "geocoding_cache.csv"
            cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
        except Exception:
            cache_df = pd.DataFrame()
        try:
            from .routes import annotate_cluster_labels

            clusters_lab = annotate_cluster_labels(clusters.head(50), precision=0.003, cache_df=cache_df)
            # Persist/merge cache
            if not clusters_lab.empty:
                merged_cache = (
                    pd.concat([cache_df, clusters_lab[["cluster", "label"]]], ignore_index=True)
                    .drop_duplicates(subset=["cluster"], keep="last")
                )
                merged_cache.to_csv(cache_path, index=False)
            # Bring labels into perf for plotting
            if not perf.empty and not clusters_lab.empty:
                perf = perf.merge(
                    clusters_lab[["cluster", "label"]], on="cluster", how="left"
                )
        except Exception:
            pass
        clusters.to_csv(out_csv / "route_start_clusters.csv", index=False)
        perf.to_csv(out_csv / "route_cluster_perf.csv", index=False)
        viz.bar_route_clusters(perf, out_plots / "route_clusters.png")

    dev = device_bias(workouts)
    if not dev.empty:
        dev.to_csv(out_csv / "device_bias.csv", index=False)
        viz.bar_device_bias(dev, out_plots / "device_bias.png")
        from .devices import device_bucket_matrix

        mat = device_bucket_matrix(workouts)
        if not mat.empty:
            mat.to_csv(out_csv / "device_bucket_matrix.csv")
        viz.heatmap_matrix(
            mat,
            "Device × Bucket Pace Delta",
            "Bucket",
            "Device",
            out_plots / "device_bucket_heatmap.png",
        )

    prs = pr_by_distance_bucket(workouts)
    prs.to_csv(out_csv / "prs_by_bucket.csv", index=False)
    viz.timeline_prs(prs, out_plots / "prs_timeline.png")
    # Common race PRs
    from .prs import pr_common_races

    pr_races = pr_common_races(workouts)
    pr_races.to_csv(out_csv / "prs_common_races.csv", index=False)
    if not pr_races.empty:
        viz.timeline_prs(
            pr_races.rename(columns={"race": "bucket"}),
            out_plots / "prs_common_timeline.png",
        )

    # Outliers
    ztab = pace_zscores_by_bucket(workouts)
    ztab.to_csv(out_csv / "outliers_zscores.csv", index=False)
    resid = efficiency_residuals(workouts)
    resid.to_csv(out_csv / "efficiency_residuals.csv", index=False)
    maha = multivariate_outliers(workouts)
    maha.to_csv(out_csv / "multivariate_outliers.csv", index=False)
    # Correlation matrix
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
        # Spearman via ranked Pearson to avoid scipy dependency
        corr = workouts[num_cols].rank(pct=True).corr(method="pearson")
        corr.to_csv(out_csv / "correlation_matrix.csv")
        viz.heatmap_matrix(
            corr, "Correlation Matrix", "Features", "Features", out_plots / "correlation_matrix.png"
        )
        # Also compute correlation of each feature with pace for clarity
        if "pace_min_per_mile_eff" in workouts:
            target = pd.to_numeric(workouts["pace_min_per_mile_eff"], errors="coerce")
            feats = [c for c in num_cols if c != "pace_min_per_mile_eff"]
            rows = []
            for c in feats:
                s = pd.to_numeric(workouts[c], errors="coerce")
                if s.notna().sum() >= 10 and target.notna().sum() >= 10:
                    coef = target.rank(pct=True).corr(s.rank(pct=True), method="pearson")
                    rows.append({"feature": c, "spearman_vs_pace": float(coef)})
            if rows:
                pc = pd.DataFrame(rows).sort_values("spearman_vs_pace")
                pc.to_csv(out_csv / "pace_correlations.csv", index=False)
                viz.bar(
                    pc["feature"].tolist(),
                    pc["spearman_vs_pace"].tolist(),
                    "Feature correlation with pace (Spearman)",
                    "Feature",
                    "ρ vs pace",
                    out_plots / "pace_correlations.png",
                )

    # Load (weekly + A:C) from workout_stats if edwards/TRIMP available
    wl = weekly_load(workouts)
    if not wl.empty:
        wl.to_csv(out_csv / "weekly_load.csv", index=False)
        acr = acute_chronic_ratio(wl)
        acr.to_csv(out_csv / "acute_chronic_ratio.csv", index=False)

    # Weather (optional)
    if with_weather and routes is not None and not routes.empty:
        ww = annotate_weather(workouts, routes)
        if not ww.empty:
            wj = workouts.merge(ww, on="workout_id", how="left")
            wj = add_effective_fields(wj)
            wj.to_csv(out_csv / "workouts_weather.csv", index=False)
            we = weather_effects(wj)
            we.to_csv(out_csv / "weather_effects.csv", index=False)

    # Summary markdown
    ann = annual_report(workouts, routes)
    mon = monthly_report(workouts)
    write_summary_md(out_path / "summary.md", ann, mon, prs)
    write_plots_catalog(out_plots)
    write_markdown_suite(out_path)

    # HR-dependent and elevation-dependent extras
    try:
        hrs = _read_df(in_path, "hr_samples")
    except FileNotFoundError:
        hrs = pd.DataFrame()
    try:
        daily = _read_df(in_path, "daily_summary")
    except FileNotFoundError:
        daily = pd.DataFrame()
    if not hrs.empty:
        from .intensity import estimate_hrmax, estimate_hrrest, zones_trimp_for_workouts

        hrmax = estimate_hrmax(workouts, hrs)
        hrrest = estimate_hrrest(daily, hrs)
        zt = zones_trimp_for_workouts(hrs, workouts, hrmax, hrrest)
        if not zt.empty:
            wk_hr = workouts.merge(zt, on="workout_id", how="left")
            wk_hr.to_csv(out_csv / "workouts_with_hr_load.csv", index=False)
            zone_cols = ["z1_min", "z2_min", "z3_min", "z4_min", "z5_min"]
            if set(zone_cols).issubset(wk_hr.columns):
                totals = wk_hr[zone_cols].sum()
                viz.donut(
                    ["Z1", "Z2", "Z3", "Z4", "Z5"],
                    totals.tolist(),
                    "Zone Distribution",
                    out_plots / "zone_distribution_donut.png",
                )
                wk_hr["month"] = pd.to_datetime(wk_hr["start_dt_local"]).dt.strftime("%Y-%m")
                monthly = wk_hr.groupby("month")[zone_cols].sum().reset_index()
                series = {k: monthly[k] for k in zone_cols}
                viz.stack_area(
                    monthly["month"].tolist(),
                    series,
                    "Monthly Zone Minutes",
                    "Month",
                    "Minutes",
                    out_plots / "monthly_zones_stacked.png",
                )
                easy = (monthly["z1_min"] + monthly["z2_min"]) / (
                    monthly[zone_cols].sum(axis=1).replace(0, pd.NA)
                )
                pol = pd.DataFrame({"month": monthly["month"], "easy_share": easy})
                pol.to_csv(out_csv / "monthly_zone_minutes.csv", index=False)
                viz.line_monthly_miles(
                    pol.rename(columns={"easy_share": "total_miles"}),
                    out_plots / "polarization_easy_share.png",
                )
        # HR drift
        from .outliers import hr_drift as _hr

        dr = _hr(hrs, workouts)
        if not dr.empty:
            dr = workouts[["workout_id", "miles"]].merge(dr, on="workout_id", how="left")
            dr.to_csv(out_csv / "workouts_with_hr_drift.csv", index=False)
            viz.scatter(
                dr["miles"],
                dr["hr_drift_pct"],
                "HR Drift vs Distance",
                "Miles",
                "HR drift %",
                out_plots / "hr_drift_vs_distance.png",
                add_trendline=True,
                add_binned_avg=True,
            )
            viz.hist(
                dr["hr_drift_pct"],
                30,
                "HR Drift % Histogram",
                "HR drift %",
                out_plots / "hr_drift_hist.png",
            )

    # Elevation and starts
    if routes is not None and not routes.empty:
        from .routes import join_elevation as _je

        elev = _je(workouts, routes)
        if not elev.empty and "elev_per_mile" in elev:
            elev.to_csv(out_csv / "workouts_with_elevation.csv", index=False)
            viz.scatter(
                elev["elev_per_mile"],
                elev["pace_min_per_mile_eff"],
                "Pace vs Elevation Gain per Mile",
                "Elevation gain per mile (m)",
                "Pace (min/mi)",
                out_plots / "pace_vs_elev_per_mile.png",
                invert_y=True,
                add_trendline=True,
                add_binned_avg=True,
            )
            viz.hist(
                elev["elev_per_mile"],
                30,
                "Elevation Gain per Mile",
                "m gain / mile",
                out_plots / "hist_elev_per_mile.png",
            )
        if {"start_lat", "start_lon"}.issubset(routes.columns) and len(routes) >= 10:
            if 'clusters_lab' in locals() and clusters_lab is not None and not clusters_lab.empty:
                viz.scatter_start_locations_with_labels(
                    routes["start_lon"],
                    routes["start_lat"],
                    clusters_lab,
                    precision=0.003,
                    path=str(out_plots / "start_locations_scatter.png"),
                )
            else:
                viz.scatter(
                    routes["start_lon"],
                    routes["start_lat"],
                    "Start Locations",
                    "Longitude",
                    "Latitude",
                    out_plots / "start_locations_scatter.png",
                    add_trendline=False,
                    add_binned_avg=False,
                )

    # Calendar heatmaps with consistent color scale across years
    try:
        from .trends import calendar_heatmap_data
        dt_all = pd.to_datetime(workouts["start_dt_local"], utc=True)
        daily_miles_all = (
            pd.DataFrame({"date": dt_all.dt.tz_convert(None).dt.date, "miles": workouts["miles"]})
            .groupby("date")["miles"].sum()
        )
        if not daily_miles_all.empty:
            vmax = float(np.nanpercentile(daily_miles_all.values, 95))
        else:
            vmax = None
        yrs = dt_all.dt.year.dropna().astype(int).unique().tolist()
        for yr in sorted(yrs):
            mat, _, _ = calendar_heatmap_data(workouts, yr)
            viz.calendar_heatmap(mat, yr, out_plots / f"calendar_heatmap_{yr}.png", vmin=0.0, vmax=vmax)
    except Exception:
        pass

    # Health rollups and ECG
    try:
        daily = _read_df(in_path, "daily_summary")
    except FileNotFoundError:
        daily = pd.DataFrame()
    if not daily.empty:
        from .health import daily_rollups

        droll = daily_rollups(daily)
        if not droll.empty:
            droll.to_csv(out_csv / "daily_with_rollups.csv", index=False)
            # Plots
            viz.line_daily_rollup(
                droll, "rhr_7d", "Resting HR (7-day mean)", "BPM", out_plots / "rhr_7d.png"
            )
            viz.line_daily_rollup(
                droll, "hrv_7d", "HRV SDNN (7-day mean)", "ms", out_plots / "hrv_7d.png"
            )
            viz.line_daily_rollup(
                droll, "vo2_30d", "VO2max (30-day mean)", "mL/kg·min", out_plots / "vo2max_30d.png"
            )

    # ECG summary counts
    try:
        ecg = _read_df(in_path, "ecg_summary")
    except FileNotFoundError:
        ecg = pd.DataFrame()
    if not ecg.empty:
        from .health import ecg_counts

        cnt = ecg_counts(ecg)
        if not cnt.empty:
            cnt.to_csv(out_csv / "ecg_classification_counts.csv", index=False)
            viz.bar_counts(
                cnt, "classification", "count", "ECG Classification Counts", out_plots / "ecg_counts.png"
            )

    # Final pass: regenerate markdown now that all plots/tables exist
    write_markdown_suite(out_path)

    # Steps integration: produce tables/plots into unified analysis outputs
    if not daily.empty and "steps" in daily.columns:
        cfg = _StepsConfig(step_goal=10_000, stride_m=0.78)
        dsteps = _norm_steps(daily, cfg)
        # Tables → analysis/tables
        (_steps_lifetime(dsteps, cfg)).to_csv(out_csv / "steps_lifetime_summary.csv", index=False)
        (_steps_monthly(dsteps)).to_csv(out_csv / "steps_monthly.csv", index=False)
        (_steps_yearly(dsteps)).to_csv(out_csv / "steps_yearly.csv", index=False)
        (_steps_weekly(dsteps)).to_csv(out_csv / "steps_weekly.csv", index=False)
        (_steps_dist(dsteps)).to_csv(out_csv / "steps_distribution.csv", index=False)
        (_steps_dow(dsteps)).to_csv(out_csv / "steps_dayofweek.csv", index=False)
        top, low = _steps_toplow(dsteps)
        top.to_csv(out_csv / "steps_top_days.csv", index=False)
        low.to_csv(out_csv / "steps_low_days.csv", index=False)
        (_steps_streaks(dsteps, cfg)).to_csv(out_csv / "steps_streaks.csv", index=False)
        corr_df = _steps_corr(dsteps)
        if corr_df is not None and not corr_df.empty:
            corr_df.to_csv(out_csv / "steps_correlations.csv", index=False)
        # Plots → analysis/plots (no steps calendar to avoid duplicating miles calendar)
        _plot_steps_daily(dsteps, out_plots)
        _plot_steps_weekly(_steps_weekly(dsteps), out_plots)
        _plot_steps_month_goal(_steps_monthly(dsteps), cfg, out_plots)
        _plot_steps_yearly(_steps_yearly(dsteps), out_plots)
        _plot_steps_hist(dsteps, out_plots)
        _plot_steps_cdf(dsteps, out_plots)
        _plot_steps_dow(_steps_dow(dsteps), out_plots)
        _plot_steps_bucket(_steps_dist(dsteps), out_plots)
        _plot_steps_goalpct(_steps_weekly(dsteps), out_plots)
        for v in [
            "active_kcal",
            "exercise_min",
            "rhr_bpm",
            "vo2max_mlkgmin",
            "walking_hr_avg_bpm",
        ]:
            _plot_steps_vs(dsteps, v, out_plots)
        # Regenerate markdown to include steps section
        write_markdown_suite(out_path)


@click.group()
def analyze() -> None:
    """Analysis CLI."""


@analyze.command("all")
@click.option("--in", "indir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out", "outdir", required=True, type=click.Path(file_okay=False))
@click.option("--tz", default="America/Denver", show_default=True)
def analyze_all(indir: str, outdir: str, tz: str) -> None:
    run_all(indir, outdir, tz)


if __name__ == "__main__":  # pragma: no cover
    analyze()
