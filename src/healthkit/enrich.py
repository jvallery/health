from __future__ import annotations

import pandas as pd

from analysis.common import add_effective_fields
from analysis.weather import annotate_weather, weather_effects


def join_weather(
    workouts: pd.DataFrame, routes: pd.DataFrame, *, cache_dir: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if routes is None or routes.empty:
        return pd.DataFrame(), pd.DataFrame()
    # analysis.weather.annotate_weather currently ignores cache_dir
    ww = annotate_weather(workouts, routes)
    if ww.empty:
        return pd.DataFrame(), pd.DataFrame()
    wj = workouts.merge(ww, on="workout_id", how="left")
    wj = add_effective_fields(wj)
    we = weather_effects(wj)
    return wj, we
