from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .parsing.export_xml import parse_workouts
from .parsing.gpx import parse_gpx_dir
from .parsing.ecg_csv import parse_ecg_csv
from .transform.daily import daily_summary_stream
from .transform.routes import link_routes
from .transform.workouts import derive_workout_metrics
from .utils import io as io_utils
from .utils import time as time_utils


def _read_df(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def load_normalized(in_dir: str) -> dict[str, pd.DataFrame]:
    base = Path(in_dir)
    return {
        "workout_stats": _read_df(base, "workout_stats"),
        "routes": _read_df(base, "routes"),
        "hr_samples": _read_df(base, "hr_samples"),
        "daily_summary": _read_df(base, "daily_summary"),
        "ecg_summary": _read_df(base, "ecg_summary"),
        "qa_flags": _read_df(base, "qa_flags"),
    }


def from_export(
    xml_path: str,
    routes_dir: str | None,
    ecg_dir: str | None,
    out_dir: str | None,
    tz_name: str,
    with_points: bool = False,
    with_hr_samples: bool = False,
) -> dict[str, pd.DataFrame]:
    out = Path(out_dir) if out_dir else None

    workouts = parse_workouts(xml_path)
    workouts["start_local"] = time_utils.to_local(workouts["start_utc"], tz_name)
    workouts["end_local"] = time_utils.to_local(workouts["end_utc"], tz_name)

    routes_df = pd.DataFrame()
    if routes_dir and Path(routes_dir).exists():
        routes_df, _ = parse_gpx_dir(routes_dir, with_points=with_points)
    if not routes_df.empty:
        linked = link_routes(workouts.copy(), routes_df.copy())
        workouts = linked

    stats = derive_workout_metrics(workouts, routes_df if not routes_df.empty else None)
    daily = daily_summary_stream(xml_path, tz_name)

    ecg_rows: list[dict[str, Any]] = []
    if ecg_dir and Path(ecg_dir).exists():
        for f in sorted(Path(ecg_dir).glob("*.csv")):
            row, _ = parse_ecg_csv(str(f))
            if row:
                ecg_rows.append(row)
    ecg_df = pd.DataFrame(ecg_rows) if ecg_rows else pd.DataFrame()

    if out:
        out.mkdir(parents=True, exist_ok=True)
        io_utils.write_parquet_csv(workouts, out / "workouts")
        if not routes_df.empty:
            io_utils.write_parquet_csv(routes_df, out / "routes")
        io_utils.write_parquet_csv(stats, out / "workout_stats")
        if not daily.empty:
            io_utils.write_parquet_csv(daily, out / "daily_summary")
        if not ecg_df.empty:
            io_utils.write_parquet_csv(ecg_df, out / "ecg_summary")

    return {
        "workouts": workouts,
        "routes": routes_df,
        "workout_stats": stats,
        "daily_summary": daily,
        "ecg_summary": ecg_df,
    }

