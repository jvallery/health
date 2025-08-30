from __future__ import annotations

import math

import numpy as np
import pandas as pd


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_m * c


def haversine_series(lat: pd.Series, lon: pd.Series) -> pd.Series:
    lat1 = lat.shift(1)
    lon1 = lon.shift(1)
    lat2 = lat
    lon2 = lon
    # Vectorized using numpy for speed
    earth_radius_m = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = earth_radius_m * c
    d.iloc[0] = 0.0
    return d.fillna(0.0)


def elevation_gain_loss(elevations: np.ndarray, threshold: float = 3.0) -> tuple[float, float]:
    if elevations.size == 0:
        return 0.0, 0.0
    diffs = np.diff(elevations)
    gains = diffs[diffs > threshold].sum() if diffs.size else 0.0
    losses = -diffs[diffs < -threshold].sum() if diffs.size else 0.0
    return float(gains), float(losses)
