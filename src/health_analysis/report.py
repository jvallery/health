from __future__ import annotations

from pathlib import Path

import pandas as pd


def _link_list(dir_path: Path, sub: str, names: list[str]) -> list[str]:
    lines: list[str] = []
    p = dir_path / sub
    for nm in names:
        f = p / nm
        if f.exists():
            lines.append(f"- {nm}: ./{sub}/{nm}")
    return lines


def write_markdown(
    out_dir: str | Path,
    body_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    years: list[int],
    *,
    weight_goal_lb: float | None,
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    start = ""
    end = ""
    if not vitals_df.empty and "date" in vitals_df:
        start = pd.to_datetime(vitals_df["date"]).min().date().isoformat()
        end = pd.to_datetime(vitals_df["date"]).max().date().isoformat()
    elif not energy_df.empty and "date" in energy_df:
        start = pd.to_datetime(energy_df["date"]).min().date().isoformat()
        end = pd.to_datetime(energy_df["date"]).max().date().isoformat()

    lines: list[str] = []
    lines.append(f"# Health Report ({start} → {end})")
    lines.append("")

    # Tables
    lines.append("## Tables")
    lines += _link_list(
        out,
        "tables",
        [
            "body_weight_daily.csv",
            "body_weight_monthly.csv",
            "body_fat_daily.csv",
            "bmi_daily.csv",
            "resting_hr_daily.csv",
            "rhr_monthly.csv",
            "hrv_daily.csv",
            "vo2max_daily.csv",
            "bp_daily.csv",
            "spo2_daily.csv",
            "respiratory_daily.csv",
            "active_vs_basal_daily.csv",
            "active_vs_basal_monthly.csv",
            "sleep_daily.csv",
            "sleep_efficiency.csv",
            "hr_alerts.csv",
            "audio_exposure.csv",
            "hrv_low_flags.csv",
        ],
    )
    lines.append("")

    # Plots
    lines.append("## Plots")
    pdir = out / "plots"
    plot_names = [
        # Body
        "weight_trend.png",
        "weight_goal.png",
        "bodyfat_vs_weight.png",
        "bmi_trend.png",
        # Vitals
        "resting_hr_trend.png",
        "hrv_trend.png",
        "vo2max_trend.png",
        "bp_trend.png",
        "spo2_hist.png",
        "respiratory_trend.png",
        # Energy
        "energy_daily_stacked.png",
        "energy_monthly_stacked.png",
        # Sleep
        "sleep_duration.png",
        "sleep_efficiency.png",
        # Alerts
        "hr_alerts_timeline.png",
        "audio_exposure_weekly.png",
    ]
    for nm in plot_names:
        f = pdir / nm
        if f.exists():
            lines.append(f"### {nm}")
            lines.append(f"![{nm}](./plots/{nm})")

    for y in years:
        c = pdir / f"sleep_calendar_{y}.png"
        if c.exists():
            lines.append(f"### sleep_calendar_{y}.png")
            lines.append(f"![sleep_calendar_{y}](./plots/sleep_calendar_{y}.png)")

    p = out / "report_health.md"
    p.write_text("\n".join(lines))
    return str(p)

