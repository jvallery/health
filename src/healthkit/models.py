from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RouteSummary:
    route_id: str
    file: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    points: int
    distance_m: float
    moving_time_sec: float
    elev_gain_m: float
    elev_loss_m: float
