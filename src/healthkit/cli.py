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


@main.command()
@click.option(
    "--in",
    "in_dir",
    type=click.Path(exists=True, file_okay=False),
    default="out",
    show_default=True,
)
@click.option(
    "--reports",
    "reports_dir",
    type=click.Path(file_okay=False),
    default="out/reports",
    show_default=True,
)
@click.option("--with-weather/--no-weather", default=False, show_default=True)
@click.option("--tz", "user_tz", default="America/Denver", show_default=True)
def analyze(in_dir: str, reports_dir: str, with_weather: bool, user_tz: str) -> None:
    """Run full analysis suite (trends, plots, reports)."""
    from analysis.cli import run_all

    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    run_all(in_dir, reports_dir, tz=user_tz, with_weather=with_weather)


# Wire in extended health analytics (full metrics suite)
try:
    from health_analysis.cli import analyze_health as analyze_health_cmd  # type: ignore

    main.add_command(analyze_health_cmd)
except Exception:
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
