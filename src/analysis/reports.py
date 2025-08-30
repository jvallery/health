from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import add_effective_fields
from .devices import device_bias
from .routes import route_cluster_perf
from .trends import monthly_perf


def _fmt_minutes(x: float | None) -> str:
    if x is None or pd.isna(x):
        return ""
    m = int(x)
    s = int(round((x - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:d}:{s:02d}"


def annual_report(
    workouts: pd.DataFrame,
    routes: pd.DataFrame | None = None,
    daily_records: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    load: pd.DataFrame | None = None,
    year: int | None = None,
) -> dict:
    w = add_effective_fields(workouts.copy())
    w["start_utc"] = pd.to_datetime(w.get("start_utc", w.get("start_local")), utc=True)
    if year is None:
        year = int(w["start_utc"].dt.year.max())
    wy = w[w["start_utc"].dt.year == year]
    mp = monthly_perf(wy)
    total_miles = float(mp["total_miles"].sum()) if not mp.empty else 0.0
    avg_pace = float(mp["avg_pace"].mean()) if not mp.empty else float("nan")
    best_month = mp.sort_values("avg_pace").iloc[0]["month"] if not mp.empty else ""
    rep = {
        "year": year,
        "workouts": int(len(wy)),
        "total_miles": total_miles,
        "avg_pace_min_per_mile": avg_pace,
        "avg_pace_str": _fmt_minutes(avg_pace),
        "best_month": best_month,
    }
    return rep


def monthly_report(workouts: pd.DataFrame, month: str | None = None) -> dict:
    w = add_effective_fields(workouts.copy())
    dt = pd.to_datetime(w.get("start_local", w.get("start_utc")), utc=True)
    w["month"] = dt.dt.to_period("M").astype(str)
    if month is None:
        month = str(w["month"].max())
    wm = w[w["month"] == month]
    miles = float(wm["miles"].sum()) if not wm.empty else 0.0
    pace = float(wm["pace_min_per_mile_eff"].median()) if not wm.empty else float("nan")
    return {
        "month": month,
        "workouts": int(len(wm)),
        "miles": miles,
        "median_pace": pace,
        "median_pace_str": _fmt_minutes(pace),
    }


def route_report(workouts: pd.DataFrame, routes: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    perf = route_cluster_perf(workouts, routes)
    if perf.empty:
        return perf
    if "n" in perf.columns:
        top = perf.groupby("cluster")["n"].sum()
    else:
        top = perf.groupby("cluster")["median_pace"].count()
    top_clusters = set(top.sort_values(ascending=False).head(top_n).index)
    return perf[perf["cluster"].isin(top_clusters)].copy()


def environmental_report(workouts_weather: pd.DataFrame) -> pd.DataFrame:
    # The caller should pass weather_effects() or annotated dataset
    return workouts_weather.copy()


def device_report(workouts: pd.DataFrame) -> pd.DataFrame:
    return device_bias(workouts)


def injury_risk_report(load_df: pd.DataFrame, outliers_df: pd.DataFrame) -> dict:
    high_ac = 0
    if load_df is not None and not load_df.empty and "ac_ratio" in load_df:
        high_ac = int((load_df["ac_ratio"] > 1.5).sum())
    struggles = 0
    if outliers_df is not None and not outliers_df.empty and "is_struggle" in outliers_df:
        struggles = int(outliers_df["is_struggle"].sum())
    return {"weeks_high_ac_ratio": high_ac, "struggle_runs": struggles}


def write_summary_md(path: str, annual: dict, monthly: dict, prs_df: pd.DataFrame) -> None:
    p = Path(path)
    lines = []
    lines.append("# Training Summary\n")
    lines.append(f"## Year {annual.get('year')}\n")
    lines.append(f"- Workouts: {annual.get('workouts')}")
    lines.append(f"- Miles: {annual.get('total_miles'):.1f}")
    lines.append(f"- Avg pace: {annual.get('avg_pace_str')} min/mi")
    lines.append(f"- Best month: {annual.get('best_month')}")
    lines.append(f"## Month {monthly.get('month')}\n")
    lines.append(f"- Workouts: {monthly.get('workouts')}")
    lines.append(f"- Miles: {monthly.get('miles'):.1f}")
    lines.append(f"- Median pace: {monthly.get('median_pace_str')} min/mi")
    if prs_df is not None and not prs_df.empty:
        lines.append("## Recent PRs\n")
        head = prs_df.sort_values("pace_min_per_mile").head(10)
        for _, r in head.iterrows():
            lines.append(
                f"- {r['bucket']} mi on {r['date']}: {_fmt_minutes(r['pace_min_per_mile'])} min/mi"
            )
        lines.append("")
    p.write_text("\n".join(lines))


def write_plots_catalog(plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    catalog = plots_dir / "README.md"
    order = [
        ("monthly_miles_and_runs.png", "Monthly miles (line) and number of runs (bars)."),
        ("monthly_km_and_runs.png", "Monthly kilometers and number of runs."),
        ("monthly_miles_and_runs_last24.png", "Last 24 months: miles and runs."),
        ("monthly_pace.png", "Average and median monthly pace (lower is faster)."),
        ("monthly_miles.png", "Monthly total mileage."),
        ("efficiency_trend.png", "Beats-per-mile rolling median (30 runs)."),
        ("pace_by_season.png", "Pace distribution by season (DJF/MAM/JJA/SON)."),
        ("miles_by_season.png", "Median miles by season."),
        ("dow_hour_heatmap.png", "Heatmap of workouts by day-of-week × hour."),
        ("device_bias.png", "Device pace delta vs baseline by distance bucket."),
        ("route_clusters.png", "Top route clusters by median pace."),
        ("prs_timeline.png", "Personal records timeline."),
    ]
    lines = ["# Plots Catalog", ""]
    for fname, desc in order:
        f = plots_dir / fname
        if f.exists():
            lines.append(f"## {fname}")
            lines.append(desc)
            lines.append("")
            lines.append(f"![{fname}](./{fname})")
            lines.append("")
    catalog.write_text("\n".join(lines))


def _md_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "_No data_"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown_suite(analysis_dir: Path) -> None:
    base = Path(analysis_dir)
    tables = base / "tables"
    base.mkdir(parents=True, exist_ok=True)

    def read_csv(name: str) -> pd.DataFrame:
        p = tables / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Navigation links
    nav = (
        "[Index](./index.md) | "
        "[Trends](./trends.md) | "
        "[Devices](./devices.md) | "
        "[Performance](./performance.md) | "
        "[HR & Load](./hr_load.md) | "
        "[Elevation & Routes](./elevation_routes.md) | "
        "[Calendar](./calendar.md) | "
        "[Health](./health.md)"
    )

    # Index
    index = base / "index.md"
    # Extended helper copy per plot (what/how/data/why)
    explain: dict[str, list[str]] = {
        "plots/monthly_pace.png": [
            "What: Average and median effective pace per month (lower is faster).",
            "How: Effective pace = moving_time ÷ miles; monthly aggregation then 3‑month rolling median overlay.",
            "Data: workout_stats with GPX overrides; timezone‑normalized to America/Denver.",
            "Why: Separates central tendency (median) from skew (average) and shows long‑term trend.",
        ],
        "plots/monthly_miles.png": [
            "What: Total miles per month with 3‑month rolling average.",
            "How: Sum of miles per calendar month in local time.",
            "Data: workout_stats normalized to start_dt_local.",
            "Why: Highlights consistency, base building, and ramp rates.",
        ],
        "plots/monthly_miles_and_runs.png": [
            "What: Miles line overlaid with number of workouts (bars).",
            "How: Monthly aggregation; bars use a lighter blue; miles include a 3‑month rolling line.",
            "Data: All run/walk/treadmill workouts merged.",
            "Why: Volume and frequency together better indicate training load/consistency.",
        ],
        "plots/efficiency_trend.png": [
            "What: Beats‑per‑mile (BPMile) monthly median; lower is better.",
            "How: BPMile = avg_hr_bpm × pace(min/mi); HR backfilled from HR samples when missing; 3‑point rolling median.",
            "Data: workout_stats + hr_samples (if available).",
            "Why: Tracks aerobic efficiency independent of distance.",
        ],
        "plots/pace_by_season.png": [
            "What: Pace distribution by meteorological season (DJF/MAM/JJA/SON).",
            "How: Boxplot of effective pace per season.",
            "Data: All workouts with valid pace.",
            "Why: Surfaces seasonal effects (heat/cold) on pacing.",
        ],
        "plots/miles_by_season.png": [
            "What: Median miles per workout by season.",
            "How: Per‑season median of distance.",
            "Data: All workouts.",
            "Why: Shows seasonal distance habits.",
        ],
        "plots/dow_hour_heatmap.png": [
            "What: Heatmap of workout counts by day × hour.",
            "How: Crosstab on local day‑of‑week and hour.",
            "Data: start_dt_local from workouts.",
            "Why: Reveals routine windows for training.",
        ],
        "plots/hist_distance.png": [
            "What: Histogram of workout distance (miles).",
            "How: 30 equal‑width bins.",
            "Data: Effective distance (GPX distance where available).",
            "Why: Identifies typical routes and long‑run frequency.",
        ],
        "plots/hist_pace.png": [
            "What: Histogram of effective pace.",
            "How: 30 bins; minutes per mile.",
            "Data: Effective pace derived from moving time and distance.",
            "Why: Gauges speed distribution and consistency.",
        ],
        "plots/device_bias.png": [
            "What: Device × distance‑bucket pace delta vs overall baseline.",
            "How: Per bucket median pace per device minus bucket median across all devices.",
            "Data: Source name + hardware parsed from Apple device string; includes Peloton/treadmill if present.",
            "Why: Detects systematic device measurement differences.",
        ],
        "plots/device_bucket_heatmap.png": [
            "What: Heatmap of device bias across distance buckets.",
            "How: Matrix of mean deltas; blue tones darker = larger absolute delta.",
            "Data: Same as Device Bias.",
            "Why: Quick visual for where a device deviates.",
        ],
        "plots/route_clusters.png": [
            "What: Top start‑location clusters labeled by specific area (road/neighborhood).",
            "How: Starts grouped into ≈300m cells; reverse‑geocoded via Nominatim with cache.",
            "Data: GPX route start lat/lon.",
            "Why: Shows fastest typical areas and common starting points.",
        ],
        "plots/prs_timeline.png": [
            "What: Best efforts for distance buckets over time.",
            "How: For each bucket, the workout with minimum effective pace, annotated with mm:ss.",
            "Data: All cardio workouts with distance.",
            "Why: Tracks improvements outside formal race distances.",
        ],
        "plots/prs_common_timeline.png": [
            "What: PRs at common race distances (1mi, 5K, 10K, 10mi, Half, Marathon).",
            "How: ±5% distance filter; pick best pace per distance; scatter with labels.",
            "Data: All workouts.",
            "Why: Simple benchmark board everyone understands.",
        ],
        "plots/correlation_matrix.png": [
            "What: Spearman‑style correlations across numeric features.",
            "How: Pearson of rank‑transformed columns (no SciPy); blue palette.",
            "Data: Miles, pace, HR stats, TRIMP, zone minutes where available.",
            "Why: Spots monotonic relationships without assuming linearity.",
        ],
        "plots/pace_correlations.png": [
            "What: Each feature’s correlation with pace (min/mi).",
            "How: Spearman via ranked Pearson; sorted by coefficient.",
            "Data: Same numeric features set.",
            "Why: Prioritizes variables most tied to pace.",
        ],
        "plots/zone_distribution_donut.png": [
            "What: Share of total time in HR zones.",
            "How: Sum of zone minutes from HR samples; blue gradient donut.",
            "Data: hr_samples near workouts with estimated HRmax/HRrest.",
            "Why: Quickly inspect intensity balance.",
        ],
        "plots/monthly_zones_stacked.png": [
            "What: Monthly minutes per HR zone.",
            "How: Stacked area chart over months.",
            "Data: Zone minutes derived from HR samples.",
            "Why: Visualizes periodization and intensity mix.",
        ],
        "plots/polarization_easy_share.png": [
            "What: Easy share ((Z1+Z2)/all) per month.",
            "How: Computed from monthly zone minutes; plotted as a single line.",
            "Data: Zone minutes.",
            "Why: Proxy for polarization; higher suggests more aerobic focus.",
        ],
        "plots/hr_drift_vs_distance.png": [
            "What: HR drift % vs distance with trend and rolling average lines.",
            "How: Compare mean HR first vs second half; 100×(H2−H1)/H1; add OLS trend and rolling binned mean.",
            "Data: hr_samples aligned to workouts.",
            "Why: Drift indicates decoupling; useful for aerobic base checks.",
        ],
        "plots/hr_drift_hist.png": [
            "What: Histogram of HR drift %.",
            "How: 30 bins; percent scale.",
            "Data: Derived per‑workout drift.",
            "Why: Shows distribution and tail events.",
        ],
        "plots/pace_vs_elev_per_mile.png": [
            "What: Pace vs elevation gain per mile with trend and rolling average lines.",
            "How: Join route elevation gain to workouts; compute m/mile; overlay OLS and smoothed average.",
            "Data: workouts + routes.",
            "Why: Quantifies hill cost on pace.",
        ],
        "plots/hist_elev_per_mile.png": [
            "What: Histogram of elevation gain per mile.",
            "How: 30 bins.",
            "Data: workouts+routes.",
            "Why: Indicates how hilly typical sessions are.",
        ],
        "plots/start_locations_scatter.png": [
            "What: Scatter of route start locations.",
            "How: Lon/lat plot of GPX start points; no trendline.",
            "Data: routes.",
            "Why: Shows geographic spread and clusters.",
        ],
        "plots/rhr_7d.png": [
            "What: 7‑day mean resting HR.",
            "How: Rolling mean of daily RHR.",
            "Data: Daily summaries from Apple Health records.",
            "Why: Lower trend often signals improved fitness/recovery.",
        ],
        "plots/hrv_7d.png": [
            "What: 7‑day mean HRV (SDNN).",
            "How: Rolling mean of daily HRV SDNN.",
            "Data: Daily summaries.",
            "Why: Higher trend suggests better readiness/less stress.",
        ],
        "plots/vo2max_30d.png": [
            "What: 30‑day mean VO2max.",
            "How: Rolling mean of daily VO2max.",
            "Data: Daily summaries.",
            "Why: Slow‑moving capacity indicator.",
        ],
        "plots/ecg_counts.png": [
            "What: Counts by ECG classification.",
            "How: Robust parsing of Apple ECG summary CSVs; bar chart.",
            "Data: ecg_summary.* produced at conversion.",
            "Why: Flags potential arrhythmia prevalence over time.",
        ],
    }

    def describe(rel: str) -> str:
        bullets = explain.get(rel)
        if not bullets:
            return ""
        return "\n".join([f"- {b}" for b in bullets])

    sections = [
        (
            "Trends",
            [
                ("plots/monthly_miles_and_runs.png", "Monthly Miles and # Runs"),
                ("plots/efficiency_trend.png", "HR Efficiency Trend (BPMile roll 30)"),
                ("plots/monthly_pace.png", "Monthly Pace (avg/median)"),
            ],
        ),
        (
            "Timing",
            [
                ("plots/dow_hour_heatmap.png", "Workouts by Day of Week × Hour"),
                ("plots/hist_distance.png", "Distance Histogram"),
                ("plots/hist_pace.png", "Pace Histogram"),
            ],
        ),
        (
            "Devices & Routes",
            [
                ("plots/device_bias.png", "Device Pace Delta"),
                ("plots/device_bucket_heatmap.png", "Device × Bucket Heatmap"),
                ("plots/route_clusters.png", "Top Route Clusters"),
            ],
        ),
        (
            "Performance",
            [
                ("plots/prs_timeline.png", "PR Timeline"),
                ("plots/prs_common_timeline.png", "PRs for Common Race Distances"),
                ("plots/correlation_matrix.png", "Correlation Matrix"),
                ("plots/pace_correlations.png", "Feature Correlation vs Pace"),
            ],
        ),
        (
            "HR & Load",
            [
                ("plots/zone_distribution_donut.png", "Zone Distribution"),
                ("plots/monthly_zones_stacked.png", "Monthly Zone Minutes"),
                ("plots/polarization_easy_share.png", "Polarization (Easy Share)"),
                ("plots/hr_drift_vs_distance.png", "HR Drift vs Distance"),
                ("plots/hr_drift_hist.png", "HR Drift Histogram"),
            ],
        ),
        (
            "Elevation & Starts",
            [
                ("plots/pace_vs_elev_per_mile.png", "Pace vs Elevation Gain per Mile"),
                ("plots/start_locations_scatter.png", "Start Locations"),
            ],
        ),
        ("Calendar", []),
        (
            "Steps",
            [
                ("plots/steps_daily_rolling.png", "Daily Steps with Rolling Averages"),
                ("plots/steps_monthly_totals_and_goal_days.png", "Monthly Steps and Goal Days"),
                ("plots/steps_weekly_line.png", "Weekly Total Steps"),
            ],
        ),
        (
            "Health & ECG",
            [
                ("plots/rhr_7d.png", "RHR 7-day Mean"),
                ("plots/hrv_7d.png", "HRV 7-day Mean"),
                ("plots/vo2max_30d.png", "VO2max 30-day Mean"),
                ("plots/ecg_counts.png", "ECG Classification Counts"),
            ],
        ),
    ]
    lines = [nav, "", "# Analysis Index", ""]
    for title, imgs in sections:
        lines.append(f"## {title}")
        if title == "Calendar":
            cal_imgs = sorted((base / "plots").glob("calendar_heatmap_*.png"))
            for pth in cal_imgs:
                rel = f"plots/{pth.name}"
                caption = f"Calendar Heatmap {pth.stem.split('_')[-1]}"
                lines.append(f"### {caption}")
                lines.append(f"![{caption}](./{rel})")
                lines.append("- What: Miles per day by week.\n- How: Local dates; consistent blue scale across years.\n- Data: workout_stats miles/day (GPX‑adjusted).\n- Why: Compare streaks and rhythm year‑over‑year.")
                lines.append("")
        else:
            for rel, caption in imgs:
                p = base / rel
                if p.exists():
                    lines.append(f"### {caption}")
                    lines.append(f"![{caption}](./{rel})")
                    desc = describe(rel)
                    if desc:
                        lines.append("")
                        lines.append(desc)
                    lines.append("")
        lines.append("")
    index.write_text("\n".join(lines))

    # Trends page with tables
    trends_md = base / "trends.md"
    monthly = read_csv("monthly_perf.csv")
    weekly = read_csv("weekly_volume.csv")
    tod = read_csv("time_of_day_perf.csv")
    lines = [
        nav,
        "",
        "# Trends",
        "",
        "## Monthly Performance",
        _md_table(monthly),
        "",
        "## Weekly Volume",
        _md_table(weekly),
        "",
        "## Time of Day",
        _md_table(tod),
        "",
    ]
    trends_md.write_text("\n".join(lines))

    # Devices & Routes
    devices_md = base / "devices.md"
    dev = read_csv("device_bias.csv")
    devm = read_csv("device_bucket_matrix.csv")
    route_clusters = read_csv("route_start_clusters.csv")
    route_perf = read_csv("route_cluster_perf.csv")
    lines = [
        nav,
        "",
        "# Devices & Routes",
        "",
        "![Device Pace Delta](./plots/device_bias.png)",
        describe("plots/device_bias.png"),
        "",
        "![Device × Bucket Heatmap](./plots/device_bucket_heatmap.png)",
        describe("plots/device_bucket_heatmap.png"),
        "",
        "![Top Route Clusters](./plots/route_clusters.png)",
        describe("plots/route_clusters.png"),
        "",
        "## Device Bias",
        _md_table(dev),
        "",
        "## Device × Bucket Matrix",
        _md_table(devm),
        "",
        "## Route Start Clusters",
        _md_table(route_clusters),
        "",
        "## Route Cluster Performance",
        _md_table(route_perf),
        "",
    ]
    devices_md.write_text("\n".join(lines))

    # Performance
    perf_md = base / "performance.md"
    prs = read_csv("prs_by_bucket.csv")
    prs_races = read_csv("prs_common_races.csv")
    out_z = read_csv("outliers_zscores.csv")
    eff_res = read_csv("efficiency_residuals.csv")
    corr = read_csv("correlation_matrix.csv")
    pace_corr = read_csv("pace_correlations.csv")
    lines = [
        nav,
        "",
        "# Performance & Outliers",
        "",
        "![PRs: Distance Buckets](./plots/prs_timeline.png)",
        describe("plots/prs_timeline.png"),
        "",
        "![PRs: Common Races](./plots/prs_common_timeline.png)",
        describe("plots/prs_common_timeline.png"),
        "",
        "![Feature Correlations](./plots/pace_correlations.png)",
        describe("plots/pace_correlations.png"),
        "",
        "## PRs by Bucket",
        _md_table(prs),
        "",
        "## PRs: Common Race Distances",
        _md_table(prs_races),
        "",
        "## Pace Z-scores",
        _md_table(out_z),
        "",
        "## Efficiency Residuals",
        _md_table(eff_res),
        "",
        "## Correlation Matrix",
        _md_table(corr),
        "",
    ]
    perf_md.write_text("\n".join(lines))

    # HR & Load
    hr_md = base / "hr_load.md"
    wk_hr = read_csv("workouts_with_hr_load.csv")
    monthly_z = read_csv("monthly_zone_minutes.csv")
    wl = read_csv("weekly_load.csv")
    ac = read_csv("acute_chronic_ratio.csv")
    drift = read_csv("workouts_with_hr_drift.csv")
    lines = [
        nav,
        "",
        "# Heart Rate & Load",
        "",
        "![Zone Distribution](./plots/zone_distribution_donut.png)",
        describe("plots/zone_distribution_donut.png"),
        "",
        "![Monthly Zone Minutes](./plots/monthly_zones_stacked.png)",
        describe("plots/monthly_zones_stacked.png"),
        "",
        "![Easy Share](./plots/polarization_easy_share.png)",
        describe("plots/polarization_easy_share.png"),
        "",
        "![HR Drift vs Distance](./plots/hr_drift_vs_distance.png)",
        describe("plots/hr_drift_vs_distance.png"),
        "",
        "![HR Drift Histogram](./plots/hr_drift_hist.png)",
        describe("plots/hr_drift_hist.png"),
        "",
        "## Workouts with Zones & Load",
        _md_table(wk_hr),
        "",
        "## Monthly Zones",
        _md_table(monthly_z),
        "",
        "## Weekly TRIMP",
        _md_table(wl),
        "",
        "## Acute:Chronic Ratio",
        _md_table(ac),
        "",
        "## HR Drift",
        _md_table(drift),
        "",
    ]
    hr_md.write_text("\n".join(lines))

    # Elevation & Starts
    elev_md = base / "elevation_routes.md"
    elev = read_csv("workouts_with_elevation.csv")
    lines = [
        nav,
        "",
        "# Elevation & Starts",
        "",
        "![Pace vs Elevation](./plots/pace_vs_elev_per_mile.png)",
        describe("plots/pace_vs_elev_per_mile.png"),
        "",
        "![Elevation per Mile Histogram](./plots/hist_elev_per_mile.png)",
        describe("plots/hist_elev_per_mile.png"),
        "",
        "![Start Locations](./plots/start_locations_scatter.png)",
        describe("plots/start_locations_scatter.png"),
        "",
        "## Elevation per Mile",
        _md_table(elev),
        "",
    ]
    elev_md.write_text("\n".join(lines))

    # Health & ECG
    health_md = base / "health.md"
    daily = read_csv("daily_with_rollups.csv")
    ecg = read_csv("ecg_classification_counts.csv")
    lines = [
        nav,
        "",
        "# Health & ECG",
        "",
        "![RHR 7d](./plots/rhr_7d.png)",
        describe("plots/rhr_7d.png"),
        "",
        "![HRV 7d](./plots/hrv_7d.png)",
        describe("plots/hrv_7d.png"),
        "",
        "![VO2max 30d](./plots/vo2max_30d.png)",
        describe("plots/vo2max_30d.png"),
        "",
        "![ECG Counts](./plots/ecg_counts.png)",
        describe("plots/ecg_counts.png"),
        "",
        "## Daily Rollups",
        _md_table(daily),
        "",
        "## ECG Classifications",
        _md_table(ecg),
        "",
    ]
    health_md.write_text("\n".join(lines))

    # Calendar page (optional)
    cal_md = base / "calendar.md"
    cal_lines = [
        nav,
        "",
        "# Calendar",
        "",
    ]
    for y in range(2005, 2100):  # include only those that exist
        f = base / f"plots/calendar_heatmap_{y}.png"
        if f.exists():
            cal_lines.append(f"## {y}")
            cal_lines.append(f"![Calendar {y}](./plots/calendar_heatmap_{y}.png)")
            cal_lines.append(
                "- What: Miles per day by week.\n- How: Local dates aggregated; Monday‑start week grid.\n- Data: workout_stats (miles/day).\n- Why: See streaks, gaps, and weekly rhythm."
            )
            cal_lines.append("")
    if len(cal_lines) > 4:
        cal_md.write_text("\n".join(cal_lines))

    # Steps page
    steps_md = base / "steps.md"
    # Read steps tables if present
    s_life = read_csv("steps_lifetime_summary.csv")
    s_mon = read_csv("steps_monthly.csv")
    s_year = read_csv("steps_yearly.csv")
    s_week = read_csv("steps_weekly.csv")
    s_dow = read_csv("steps_dayofweek.csv")
    s_dist = read_csv("steps_distribution.csv")
    s_top = read_csv("steps_top_days.csv")
    s_low = read_csv("steps_low_days.csv")
    s_strk = read_csv("steps_streaks.csv")
    s_corr = read_csv("steps_correlations.csv")
    lines = [
        nav,
        "",
        "# Steps",
        "",
        "![Daily Steps](./plots/steps_daily_rolling.png)",
        "",
        "![Monthly Steps and Goal Days](./plots/steps_monthly_totals_and_goal_days.png)",
        "",
        "![Weekly Steps](./plots/steps_weekly_line.png)",
        "",
        "## Lifetime Summary",
        _md_table(s_life),
        "",
        "## Monthly",
        _md_table(s_mon),
        "",
        "## Yearly",
        _md_table(s_year),
        "",
        "## Weekly",
        _md_table(s_week),
        "",
        "## Day of Week",
        _md_table(s_dow),
        "",
        "## Distribution",
        _md_table(s_dist),
        "",
        "## Top Days",
        _md_table(s_top),
        "",
        "## Low Days",
        _md_table(s_low),
        "",
        "## Streaks",
        _md_table(s_strk),
        "",
        "## Correlations",
        _md_table(s_corr),
        "",
    ]
    steps_md.write_text("\n".join(lines))

    # Prepend nav to summary.md if present
    summary = base / "summary.md"
    if summary.exists():
        content = summary.read_text()
        summary.write_text(nav + "\n\n" + content)
