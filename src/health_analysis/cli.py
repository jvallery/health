from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from .common import HealthConfig, parse_records, daily_aggregates, join_daily_frames
from . import tables as T
from . import plots as P
from .report import write_markdown


def _enough(df: pd.DataFrame, col: str, n: int = 30) -> bool:
    try:
        return int(pd.to_numeric(df[col], errors="coerce").notna().sum()) >= int(n)
    except Exception:
        return False


@click.command(name="analyze-health")
@click.option("--xml", "xml_path", type=click.Path(exists=True, dir_okay=False), default="apple_health_export/export.xml", show_default=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="out/health", show_default=True)
@click.option("--tz", "tz_name", default="America/Denver", show_default=True)
@click.option("--weight-goal", "weight_goal_lb", type=float, default=None, help="Optional weight goal (lb) for overlay")
def analyze_health(xml_path: str, out_dir: str, tz_name: str, weight_goal_lb: float | None) -> None:
    out = Path(out_dir)
    tdir = out / "tables"
    pdir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    # Wire config
    cfg = HealthConfig(tz=tz_name, weight_goal_lb=weight_goal_lb, out_weight_lb=True)

    # Whitelist types to parse
    types = [
        # Body
        "BodyMass",
        "BodyFatPercentage",
        "BodyMassIndex",
        # Vitals
        "RestingHeartRate",
        "WalkingHeartRateAverage",
        "HeartRateVariabilitySDNN",
        "VO2Max",
        "BloodPressureSystolic",
        "BloodPressureDiastolic",
        "OxygenSaturation",
        "RespiratoryRate",
        # Energy & steps & exercise
        "ActiveEnergyBurned",
        "BasalEnergyBurned",
        "AppleExerciseTime",
        "StepCount",
        # Sleep & events
        "SleepAnalysis",
        "HighHeartRateEvent",
        "LowHeartRateEvent",
        # Audio
        "HeadphoneAudioExposure",
        "EnvironmentalAudioExposure",
    ]

    click.echo("[health] Parsing records (stream)")
    recs = parse_records(xml_path, type_whitelist=types)
    click.echo(f"[health] Records loaded: {len(recs):,}")

    click.echo("[health] Building daily aggregates")
    daily = daily_aggregates(recs, tz=tz_name)
    body_df = daily.get("body", pd.DataFrame())
    vitals_df = daily.get("vitals", pd.DataFrame())
    energy_df = daily.get("energy", pd.DataFrame())
    sleep_df = daily.get("sleep", pd.DataFrame())
    alerts_df = daily.get("alerts", pd.DataFrame())
    audio_df = daily.get("audio", pd.DataFrame())

    # Combined for derived correlations/years list
    combined = join_daily_frames([body_df, vitals_df, energy_df, sleep_df])
    years = (
        sorted(pd.to_datetime(combined["date"]).dt.year.dropna().astype(int).unique().tolist()) if not combined.empty else []
    )

    # 2) Tables
    if not body_df.empty:
        T.body_weight_daily(body_df, tdir, to_lb=cfg.out_weight_lb)
        T.body_weight_monthly(body_df, tdir, to_lb=cfg.out_weight_lb)
        if "body_fat_frac" in body_df:
            T.body_fat_daily(body_df, tdir)
        if "bmi" in body_df:
            T.bmi_daily(body_df, tdir)

    if not vitals_df.empty:
        if "rhr_bpm" in vitals_df:
            T.resting_hr_daily(vitals_df, tdir)
            T.resting_hr_monthly(vitals_df, tdir)
        if "hrv_sdnn_ms" in vitals_df:
            T.hrv_daily(vitals_df, tdir)
        if "vo2max_mlkgmin" in vitals_df:
            T.vo2max_daily(vitals_df, tdir)
        if "bp_systolic_mmhg" in vitals_df or "bp_diastolic_mmhg" in vitals_df:
            T.bp_daily(vitals_df, tdir)
        if "spo2_frac" in vitals_df:
            T.spo2_daily(vitals_df, tdir)
        if "resp_rate_bpm" in vitals_df:
            T.respiratory_daily(vitals_df, tdir)

    if not energy_df.empty:
        T.energy_daily(energy_df, tdir)
        T.energy_monthly(energy_df, tdir)

    if not sleep_df.empty:
        T.sleep_daily(sleep_df, tdir)
        if "sleep_efficiency" in sleep_df:
            eff = T.sleep_efficiency_daily(sleep_df, tdir)
            if not eff.empty and (eff["sleep_efficiency"] > 1).any():
                click.echo("[WARN] sleep efficiency > 1.0 found; check SleepAnalysis parsing")

    if not alerts_df.empty:
        T.hr_alerts_monthly(alerts_df, tdir)
    if not audio_df.empty:
        T.audio_exposure_weekly(audio_df, tdir)

    # Derived
    slope = T.weight_slope(body_df, to_lb=True) if not body_df.empty else None
    r_corr, n_corr = T.rhr_vo2_corr(vitals_df) if not vitals_df.empty else (None, 0)
    if not vitals_df.empty and "hrv_sdnn_ms" in vitals_df:
        T.hrv_low_flags(vitals_df, out=tdir)

    # 3) Plots
    if not body_df.empty:
        if _enough(body_df, "weight", n=7):
            P.weight_trend(body_df.assign(weight_lb=body_df["weight"] / 0.45359237), pdir) if cfg.out_weight_lb else P.weight_trend(body_df, pdir)
        if cfg.weight_goal_lb is not None and _enough(body_df, "weight", n=7):
            P.weight_goal(body_df.assign(weight_lb=body_df["weight"] / 0.45359237), cfg.weight_goal_lb, pdir)
        if "body_fat_frac" in body_df and _enough(body_df.dropna(subset=["body_fat_frac"]), "body_fat_frac", n=10):
            P.bodyfat_vs_weight(body_df.assign(weight_lb=body_df["weight"] / 0.45359237) if cfg.out_weight_lb else body_df, pdir)
        if "bmi" in body_df and _enough(body_df, "bmi", n=7):
            P.bmi_trend(body_df, pdir)

    if not vitals_df.empty:
        if "rhr_bpm" in vitals_df and _enough(vitals_df, "rhr_bpm", n=30):
            P.resting_hr_trend(vitals_df, pdir)
        if "hrv_sdnn_ms" in vitals_df and _enough(vitals_df, "hrv_sdnn_ms", n=30):
            P.hrv_trend(vitals_df, pdir)
        if "vo2max_mlkgmin" in vitals_df and _enough(vitals_df, "vo2max_mlkgmin", n=10):
            P.vo2max_trend(vitals_df, pdir)
        if ("bp_systolic_mmhg" in vitals_df or "bp_diastolic_mmhg" in vitals_df) and len(vitals_df) >= 3:
            P.bp_trend(vitals_df, pdir)
        if "spo2_frac" in vitals_df and _enough(vitals_df, "spo2_frac", n=20):
            P.spo2_hist(vitals_df.assign(spo2_pct=vitals_df["spo2_frac"] * 100.0), pdir)
        if "resp_rate_bpm" in vitals_df and _enough(vitals_df, "resp_rate_bpm", n=10):
            P.respiratory_trend(vitals_df, pdir)

    if not energy_df.empty and ("active_kcal" in energy_df or "basal_kcal" in energy_df):
        P.energy_daily_stacked(energy_df, pdir)
        monthly = T.energy_monthly(energy_df)
        if not monthly.empty:
            P.energy_monthly_stacked(monthly, pdir)

    if not sleep_df.empty:
        if ("sleep_in_bed_min" in sleep_df or "sleep_asleep_min" in sleep_df) and len(sleep_df) >= 7:
            P.sleep_duration(sleep_df, pdir)
        if "sleep_efficiency" in sleep_df and _enough(sleep_df, "sleep_efficiency", n=14):
            P.sleep_efficiency(sleep_df, pdir)
        # calendar per year
        if "sleep_asleep_min" in sleep_df:
            for y in years:
                P.sleep_calendar(sleep_df, y, pdir, col="sleep_asleep_min")

    if not alerts_df.empty:
        P.hr_alerts_timeline(alerts_df, pdir)
    if not audio_df.empty:
        P.audio_exposure_weekly(audio_df, pdir)

    # 4) Report
    report_path = write_markdown(
        out,
        body_df,
        vitals_df,
        energy_df,
        sleep_df,
        alerts_df,
        audio_df,
        years,
        weight_goal_lb=cfg.weight_goal_lb,
    )
    click.echo(f"[health] Report → {report_path}")

