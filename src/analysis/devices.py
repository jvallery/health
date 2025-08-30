from __future__ import annotations

import pandas as pd

from .common import add_effective_fields


def _device_key(source_name: pd.Series, device_str: pd.Series) -> pd.Series:
    """Return concise make/model label like "Jason's Apple Watch (Watch6,4)".

    - Prefers the human-readable `source_name` (e.g., "Jason’s Apple Watch").
    - Extracts hardware code from `device` string (e.g., "hardware:Watch6,4").
    - Normalizes curly apostrophes to straight for readability.
    """
    import re

    def norm_apostrophes(s: str) -> str:
        return s.replace("’", "'")

    name = source_name.fillna("").astype(str)
    # Use entire source name if it contains "Apple Watch"; otherwise keep first 3 tokens
    def pick_name(x: str) -> str:
        x = re.sub(r"\s+", " ", x).strip()
        if "apple watch" in x.lower():
            return norm_apostrophes(x)
        return norm_apostrophes(" ".join(x.split()[:3]))

    name = name.apply(pick_name)

    dev = device_str.fillna("").astype(str)
    # Try multiple patterns for hardware identifier
    hw = (
        dev.str.extract(r"(?i)hardware\s*:?\s*([A-Za-z0-9,]+)")[0]
        .fillna(dev.str.extract(r"(?i)\b(Watch\d+,\d+)\b")[0])
        .fillna("")
        .str.replace(r"[,)\s]+$", "", regex=True)
    )
    label = name
    label = label.where(hw.eq(""), label + " (" + hw + ")")
    # Avoid overly long labels but keep owner + model
    return label.str.slice(0, 48)


def device_bias(workouts: pd.DataFrame) -> pd.DataFrame:
    if workouts.empty:
        return pd.DataFrame()
    d = add_effective_fields(workouts.copy())
    d["miles_bucket"] = pd.cut(
        d["miles"],
        bins=[0, 2, 3, 4, 5, 6, 8, 13, 20, 100],
        labels=["<2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-13", "13-20", "20+"],
        right=False,
    )
    # Canonical device name with hardware code
    d["device_key"] = _device_key(d.get("source_name"), d.get("device"))
    baseline = d.groupby("miles_bucket")["pace_min_per_mile_eff"].median().rename("baseline_pace")
    g = (
        d.groupby(["device_key", "miles_bucket"])
        .agg(median_pace=("pace_min_per_mile_eff", "median"), n=("miles", "size"))
        .reset_index()
    )
    g = g.merge(baseline, on="miles_bucket", how="left")
    g["delta_vs_all"] = g["median_pace"] - g["baseline_pace"]
    return g.sort_values(["miles_bucket", "delta_vs_all"])


def device_bucket_matrix(workouts: pd.DataFrame) -> pd.DataFrame:
    db = device_bias(workouts)
    if db.empty:
        return db
    mat = db.pivot_table(
        index="device_key", columns="miles_bucket", values="delta_vs_all", aggfunc="mean"
    ).sort_index()
    return mat
