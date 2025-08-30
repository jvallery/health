from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import io as io_utils


def assemble(
    out_dir: str,
    workouts: pd.DataFrame,
    routes: pd.DataFrame | None = None,
    points: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
) -> None:
    out = Path(out_dir)
    io_utils.write_parquet_csv(workouts, out / "workouts")
    if routes is not None and not routes.empty:
        io_utils.write_parquet_csv(routes, out / "routes")
    if points is not None and not points.empty:
        io_utils.write_parquet_csv(points, out / "route_points")
    if daily is not None and not daily.empty:
        io_utils.write_parquet_csv(daily, out / "daily_summary")
    if stats is not None and not stats.empty:
        io_utils.write_parquet_csv(stats, out / "workout_stats")
