from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Consistent palette with analysis.viz and steps.plots
BASE_BLUE = "#1f5aa6"
BLUE_LIGHT = "#4f97d7"
BLUE_DARK = "#0b3b75"
GREY_MED = "#6e7f8d"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#2b3e50",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dfe7ef",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
    }
)


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------- Body ----------


def weight_trend(body_df: pd.DataFrame, out_dir: Path) -> str | None:
    if body_df is None or body_df.empty:
        return None
    d = body_df.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    y = pd.to_numeric(d.get("weight_lb", d.get("weight")), errors="coerce")
    ax.plot(pd.to_datetime(d["date"]), y, color=BASE_BLUE, linewidth=1.3, label="daily")
    r7 = y.rolling(7, min_periods=3).mean()
    r30 = y.rolling(30, min_periods=7).mean()
    ax.plot(pd.to_datetime(d["date"]), r7, color=BLUE_LIGHT, linewidth=1.6, label="7‑day avg")
    ax.plot(pd.to_datetime(d["date"]), r30, color=GREY_MED, linewidth=1.7, label="30‑day avg")
    ax.set_title("Body Weight Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lb)" if "weight_lb" in d else "Weight (kg)")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "weight_trend.png")


def weight_goal(body_df: pd.DataFrame, goal_lb: float | None, out_dir: Path) -> str | None:
    if body_df is None or body_df.empty or goal_lb is None:
        return None
    d = body_df.copy()
    y = pd.to_numeric(d.get("weight_lb", d.get("weight")), errors="coerce")
    x = pd.to_datetime(d["date"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(x, y, color=BASE_BLUE, linewidth=1.3)
    ax.axhline(float(goal_lb), color=BLUE_DARK, linestyle="--", linewidth=1.2, label=f"Goal {goal_lb:.0f} lb")
    ax.set_title("Weight vs Goal")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lb)" if "weight_lb" in d else "Weight (kg)")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "weight_goal.png")


def bodyfat_vs_weight(body_df: pd.DataFrame, out_dir: Path) -> str | None:
    if body_df is None or body_df.empty or "body_fat_frac" not in body_df:
        return None
    d = body_df.dropna(subset=["body_fat_frac"]).copy()
    y = d["body_fat_frac"].astype(float) * 100.0
    x = pd.to_numeric(d.get("weight_lb", d.get("weight")), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x[mask], y[mask], s=14, alpha=0.6, color=BASE_BLUE)
    # trendline
    try:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        xx = np.linspace(float(x[mask].min()), float(x[mask].max()), 100)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, color=BLUE_DARK, linewidth=1.4)
    except Exception:
        pass
    ax.set_title("Body Fat vs Weight")
    ax.set_xlabel("Weight (lb)" if "weight_lb" in d else "Weight (kg)")
    ax.set_ylabel("Body fat (%)")
    return _save(fig, out_dir / "bodyfat_vs_weight.png")


def bmi_trend(body_df: pd.DataFrame, out_dir: Path) -> str | None:
    if body_df is None or body_df.empty or "bmi" not in body_df:
        return None
    d = body_df.dropna(subset=["bmi"]).copy()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(pd.to_datetime(d["date"]), d["bmi"], color=BASE_BLUE, linewidth=1.3)
    ax.set_title("BMI Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("BMI")
    return _save(fig, out_dir / "bmi_trend.png")


# ---------- Vitals ----------


def resting_hr_trend(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df is None or df.empty or "rhr_bpm" not in df:
        return None
    d = df.copy()
    y = d["rhr_bpm"].astype(float)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(pd.to_datetime(d["date"]), y, color=BASE_BLUE, linewidth=1.2, label="daily")
    ax.plot(pd.to_datetime(d["date"]), y.rolling(7, min_periods=3).mean(), color=GREY_MED, linewidth=1.6, label="7‑day avg")
    ax.set_title("Resting HR")
    ax.set_xlabel("Date")
    ax.set_ylabel("bpm")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "resting_hr_trend.png")


def hrv_trend(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df is None or df.empty or "hrv_sdnn_ms" not in df:
        return None
    d = df.copy()
    y = d["hrv_sdnn_ms"].astype(float)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(pd.to_datetime(d["date"]), y.rolling(7, min_periods=3).mean(), color=BASE_BLUE, linewidth=1.5)
    ax.set_title("HRV (SDNN) — 7‑day Rolling")
    ax.set_xlabel("Date")
    ax.set_ylabel("ms")
    return _save(fig, out_dir / "hrv_trend.png")


def vo2max_trend(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df is None or df.empty or "vo2max_mlkgmin" not in df:
        return None
    d = df.copy()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = d["vo2max_mlkgmin"].rolling(30, min_periods=5).mean()
    ax.plot(pd.to_datetime(d["date"]), y, color=BASE_BLUE, linewidth=1.5)
    ax.set_title("VO₂max — 30‑day Rolling")
    ax.set_xlabel("Date")
    ax.set_ylabel("mL/kg·min")
    return _save(fig, out_dir / "vo2max_trend.png")


def bp_trend(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df is None or df.empty:
        return None
    if "bp_systolic_mmhg" not in df and "bp_diastolic_mmhg" not in df:
        return None
    d = df.copy()
    x = pd.to_datetime(d["date"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    if "bp_systolic_mmhg" in d:
        ax.plot(x, d["bp_systolic_mmhg"], color=BASE_BLUE, linewidth=1.2, label="Systolic")
    if "bp_diastolic_mmhg" in d:
        ax.plot(x, d["bp_diastolic_mmhg"], color=GREY_MED, linewidth=1.2, label="Diastolic")
    ax.set_title("Blood Pressure")
    ax.set_xlabel("Date")
    ax.set_ylabel("mmHg")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "bp_trend.png")


def spo2_hist(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df is None or df.empty:
        return None
    col = "spo2_pct" if "spo2_pct" in df else ("spo2_frac" if "spo2_frac" in df else None)
    if col is None:
        return None
    y = df[col]
    if col == "spo2_frac":
        y = df[col].astype(float) * 100.0
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(pd.to_numeric(y, errors="coerce").dropna(), bins=30, color=BASE_BLUE)
    ax.set_title("SpO₂ Distribution")
    ax.set_xlabel("%")
    ax.set_ylabel("Count")
    return _save(fig, out_dir / "spo2_hist.png")


def respiratory_trend(df: pd.DataFrame, out_dir: Path) -> str | None:
    if df is None or df.empty or "resp_rate_bpm" not in df:
        return None
    d = df.copy()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(pd.to_datetime(d["date"]), d["resp_rate_bpm"], color=BASE_BLUE, linewidth=1.2)
    ax.set_title("Respiratory Rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("breaths/min")
    return _save(fig, out_dir / "respiratory_trend.png")


# ---------- Energy ----------


def energy_daily_stacked(eng_df: pd.DataFrame, out_dir: Path) -> str | None:
    if eng_df is None or eng_df.empty:
        return None
    cols = [c for c in ["active_kcal", "basal_kcal"] if c in eng_df]
    if not cols:
        return None
    d = eng_df.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(d["date"])  # type: ignore[arg-type]
    ax.stackplot(x, *[d[c].fillna(0).astype(float) for c in cols], labels=cols)
    ax.set_title("Daily Energy (stacked)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kcal")
    ax.legend(loc="upper left")
    return _save(fig, out_dir / "energy_daily_stacked.png")


def energy_monthly_stacked(monthly_df: pd.DataFrame, out_dir: Path) -> str | None:
    if monthly_df is None or monthly_df.empty:
        return None
    cols = [c for c in ["active_kcal", "basal_kcal"] if c in monthly_df]
    if not cols:
        return None
    d = monthly_df.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = d["month"].astype(str)
    ax.stackplot(x, *[d[c].fillna(0).astype(float) for c in cols], labels=cols)
    ax.set_title("Monthly Energy (stacked)")
    ax.set_xlabel("Month")
    ax.set_ylabel("kcal")
    ax.legend(loc="upper left")
    # reduce crowding
    if len(x) > 18:
        ticks = list(range(0, len(x), max(1, len(x) // 12)))
        ax.set_xticks([x.iloc[i] for i in ticks])
    return _save(fig, out_dir / "energy_monthly_stacked.png")


# ---------- Sleep ----------


def sleep_duration(sleep_df: pd.DataFrame, out_dir: Path) -> str | None:
    if sleep_df is None or sleep_df.empty:
        return None
    cols = [c for c in ["sleep_in_bed_min", "sleep_asleep_min"] if c in sleep_df]
    if not cols:
        return None
    d = sleep_df.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(d["date"])  # type: ignore[arg-type]
    ax.stackplot(x, *[d[c].fillna(0).astype(float) for c in cols], labels=["In bed", "Asleep"])
    ax.set_title("Sleep Duration")
    ax.set_xlabel("Date")
    ax.set_ylabel("minutes")
    ax.legend(loc="upper left")
    return _save(fig, out_dir / "sleep_duration.png")


def sleep_efficiency(sleep_df: pd.DataFrame, out_dir: Path) -> str | None:
    if sleep_df is None or sleep_df.empty or "sleep_efficiency" not in sleep_df:
        return None
    d = sleep_df.copy()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(pd.to_datetime(d["date"]), d["sleep_efficiency"], color=BASE_BLUE, linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_title("Sleep Efficiency")
    ax.set_xlabel("Date")
    ax.set_ylabel("asleep / in bed")
    return _save(fig, out_dir / "sleep_efficiency.png")


def _calendar_matrix_for_year(daily: pd.DataFrame, year: int, col: str) -> np.ndarray:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])  # ensure datetime
    dy = d[d["date"].dt.year == year]
    vals = dy.groupby(dy["date"].dt.date)[col].sum()
    import datetime as _dt

    start = _dt.date(year, 1, 1)
    start -= _dt.timedelta(days=start.weekday())  # align to Monday
    weeks = 54
    mat = np.zeros((7, weeks), dtype=float)
    for w in range(weeks):
        for dow in range(7):
            cur = start + _dt.timedelta(days=w * 7 + dow)
            mat[dow, w] = float(vals.get(cur, 0.0))
    return mat


def sleep_calendar(daily: pd.DataFrame, year: int, out_dir: Path, col: str = "sleep_asleep_min") -> str | None:
    if daily is None or daily.empty or col not in daily:
        return None
    mat = _calendar_matrix_for_year(daily, year, col)
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks([])
    ax.set_title(f"{year} Sleep Calendar (minutes asleep)")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("minutes")
    return _save(fig, out_dir / f"sleep_calendar_{year}.png")


# ---------- Alerts & Audio ----------


def hr_alerts_timeline(alerts_df: pd.DataFrame, out_dir: Path) -> str | None:
    if alerts_df is None or alerts_df.empty:
        return None
    d = alerts_df.copy()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for kind, sub in d.groupby("kind"):
        ax.plot(sub["month"], sub["count"], marker="o", linewidth=1.4, label=str(kind))
    ax.set_title("HR Alerts per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Events")
    ax.legend(loc="best", fontsize=8)
    if len(d) > 18:
        ticks = list(range(0, len(d), max(1, len(d) // 12)))
        ax.set_xticks([d["month"].unique()[i] for i in ticks])
    return _save(fig, out_dir / "hr_alerts_timeline.png")


def audio_exposure_weekly(audio_df: pd.DataFrame, out_dir: Path) -> str | None:
    if audio_df is None or audio_df.empty:
        return None
    d = audio_df.copy()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for kind, sub in d.groupby("kind"):
        ax.plot(pd.to_datetime(sub["week"]), sub["events"], linewidth=1.2, label=str(kind))
    ax.set_title("Audio Exposure — Weekly Events")
    ax.set_xlabel("Week start")
    ax.set_ylabel("Events")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out_dir / "audio_exposure_weekly.png")

