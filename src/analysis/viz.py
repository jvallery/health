from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global style: blue/grey tones, consistent fonts
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
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
    }
)

# Richer blues (avoid washed/too-light tones)
BASE_BLUE = "#1f5aa6"
BLUE_LIGHT = "#4f97d7"
BLUE_DARK = "#0b3b75"
GREY_MED = "#6e7f8d"


def _style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    invert_y: bool = False,
    rotate_x: bool = False,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    if invert_y:
        ax.invert_yaxis()
    if rotate_x:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def line_monthly_pace(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["month"], df["avg_pace"], label="Avg pace", color=BASE_BLUE, linewidth=1.8, zorder=3)
    ax.plot(
        df["month"], df["median_pace"], label="Median pace", color=GREY_MED, linestyle="--", linewidth=1.4, zorder=3
    )
    # Rolling median (3 months) overlay for miles-per-month pace
    if "median_pace" in df:
        roll = df["median_pace"].rolling(3, min_periods=1).median()
        ax.plot(df["month"], roll, color=BLUE_LIGHT, linewidth=1.4, label="Median (3-mo)", zorder=2)
    _style(ax, "Monthly Pace", "Month", "Minutes per mile", invert_y=True, rotate_x=True)
    # Reduce tick crowding
    if len(df) > 18:
        ticks = list(range(0, len(df), max(1, len(df) // 12)))
        ax.set_xticks([df["month"].iloc[i] for i in ticks])
    ax.legend()
    _save(fig, Path(path))


def line_monthly_miles(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["month"], df["total_miles"], label="Total miles", color=BASE_BLUE, linewidth=1.8, zorder=3)
    # Rolling 3-month overlay
    roll = df["total_miles"].rolling(3, min_periods=1).mean()
    ax.plot(df["month"], roll, color=GREY_MED, linewidth=1.4, label="Miles (3-mo)", zorder=2)
    _style(ax, "Monthly Mileage", "Month", "Miles", rotate_x=True)
    if len(df) > 18:
        ticks = list(range(0, len(df), max(1, len(df) // 12)))
        ax.set_xticks([df["month"].iloc[i] for i in ticks])
    _save(fig, Path(path))


def line_monthly_miles_and_runs(df: pd.DataFrame, path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df["month"], df["total_miles"], color=BASE_BLUE, label="Miles", linewidth=1.8, zorder=3)
    _style(ax1, "Monthly Miles and # Runs", "Month", "Miles", rotate_x=True)
    if len(df) > 18:
        ticks = list(range(0, len(df), max(1, len(df) // 12)))
        ax1.set_xticks([df["month"].iloc[i] for i in ticks])
    ax1.tick_params(axis="y", labelcolor=BASE_BLUE)
    ax2 = ax1.twinx()
    ax2.bar(df["month"], df["workouts"], alpha=0.3, color=BLUE_LIGHT, label="# Workouts", zorder=2)
    ax2.set_ylabel("# Workouts", color=BLUE_DARK)
    ax2.tick_params(axis="y", labelcolor=BLUE_DARK)
    # Title handled above; rotate handled by _style
    _save(fig, Path(path))


def line_monthly_km_and_runs(df: pd.DataFrame, path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df["month"], df["total_km"], color=BASE_BLUE, label="Kilometers")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Kilometers", color="#4c72b0")
    if len(df) > 18:
        ticks = list(range(0, len(df), max(1, len(df) // 12)))
        ax1.set_xticks([df["month"].iloc[i] for i in ticks])
    ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax2 = ax1.twinx()
    ax2.bar(df["month"], df["workouts"], alpha=0.25, color=BLUE_LIGHT, label="# Workouts")
    ax2.set_ylabel("# Workouts", color=BLUE_DARK)
    ax2.tick_params(axis="y", labelcolor=BLUE_DARK)
    fig.suptitle("Monthly Kilometers and # Runs")
    fig.autofmt_xdate(rotation=45)
    _save(fig, Path(path))


def line_efficiency(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["month"], df["median_beats_per_mile_roll3"], label="Median BPMile (roll3)")
    _style(ax, "HR Efficiency Trend (lower better)", "Month", "Beats per mile", rotate_x=True)
    if len(df) > 18:
        ticks = list(range(0, len(df), max(1, len(df) // 12)))
        ax.set_xticks([df["month"].iloc[i] for i in ticks])
    _save(fig, Path(path))


def box_pace_by_season(df: pd.DataFrame, path: str) -> None:
    # Expect columns: season, pace_min_per_mile_eff
    fig, ax = plt.subplots(figsize=(6, 4))
    seasons = ["DJF", "MAM", "JJA", "SON"]
    data = [df[df["season"] == s]["pace_min_per_mile_eff"].dropna() for s in seasons]
    ax.boxplot(
        data,
        labels=seasons,
        showmeans=True,
        patch_artist=True,
        boxprops=dict(facecolor=BLUE_LIGHT, edgecolor=BLUE_DARK),
        medianprops=dict(color=BLUE_DARK, linewidth=1.6),
        whiskerprops=dict(color=BLUE_DARK),
        capprops=dict(color=BLUE_DARK),
        meanprops=dict(color=GREY_MED),
        flierprops=dict(markeredgecolor=GREY_MED),
    )
    _style(ax, "Pace by Season", "Season", "Minutes per mile", invert_y=True)
    _save(fig, Path(path))


def bar_miles_by_season(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["season"], df["median_miles"], color=BASE_BLUE, zorder=3)
    _style(ax, "Median Miles by Season", "Season", "Miles")
    _save(fig, Path(path))


def bar_time_of_day(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["tod_bucket"].astype(str), df["median_pace"], color=BASE_BLUE, zorder=3)
    _style(ax, "Time-of-day Median Pace", "Bucket", "Minutes per mile", invert_y=True)
    _save(fig, Path(path))


def heatmap_dow_hour(counts: pd.DataFrame, path: str) -> None:
    # counts pivot with dow (Mon=0..Sun=6) rows and hour cols 0..23
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(counts.values, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
    _style(ax, "Workouts by Day of Week × Hour", "Hour of day", "Day of week (Mon=0)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Count")
    _save(fig, Path(path))


def bar_device_bias(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = df.copy()
    devcol = "device_key" if "device_key" in plot_df.columns else "device"
    x = plot_df[devcol].astype(str) + " | " + plot_df["miles_bucket"].astype(str)
    ax.barh(x, plot_df["delta_vs_all"], color=BASE_BLUE, zorder=3)
    _style(ax, "Device Pace Delta vs Baseline", "Δ pace (min/mi)", "Device | Bucket")
    _save(fig, Path(path))


def bar_route_clusters(
    df: pd.DataFrame, path: str, top_n: int = 10, precision: float = 0.003
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    top = (
        df.groupby(["cluster", "label"])["median_pace"].median().reset_index()
        if "label" in df.columns
        else df.groupby("cluster")["median_pace"].median().reset_index().assign(label=None)
    )
    top = top.sort_values("median_pace").head(top_n)

    def fallback_lab(c: str) -> str:
        try:
            a, b = c.split(":")
            lat = int(a) * precision
            lon = int(b) * precision
            return f"{lat:.3f}, {lon:.3f}"
        except Exception:
            return c

    labels = top.apply(
        lambda r: r["label"] if isinstance(r["label"], str) and r["label"] else fallback_lab(str(r["cluster"])) ,
        axis=1,
    )
    ax.barh(labels, top["median_pace"], color=BASE_BLUE, zorder=3)
    _style(ax, "Top Route Clusters (median pace)", "Minutes per mile", "Cluster", invert_y=True)
    _save(fig, Path(path))


def timeline_prs(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    from .common import mmss

    fig, ax = plt.subplots(figsize=(9, 5))
    d = pd.to_datetime(df["date"]) if "date" in df else pd.Series(range(len(df)))
    y = df["bucket"].astype(str)
    ax.scatter(d, y, c=BASE_BLUE, s=24, zorder=3)
    for _, r in df.iterrows():
        dx = pd.to_datetime(r["date"]) if "date" in r else 0
        ax.annotate(
            mmss(r.get("pace_min_per_mile")),
            (dx, r["bucket"]),
            fontsize=8,
            xytext=(2, 2),
            textcoords="offset points",
        )
    _style(ax, "PR Timeline", "Date", "Distance bucket")
    _save(fig, Path(path))


def hist(series: pd.Series, bins: int, title: str, xlabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna(), bins=bins, color=BASE_BLUE, edgecolor=None, zorder=3)
    _style(ax, title, xlabel, "Count")
    _save(fig, Path(path))


def donut(labels: list[str], sizes: list[float], title: str, path: str) -> None:
    if not sizes or sum(sizes) <= 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    # Blue gradient for zones
    import numpy as _np
    cols = [plt.cm.Blues(c) for c in _np.linspace(0.35, 0.85, len(sizes))]
    wedges, _ = ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=cols)
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.axis("equal")
    ax.set_title(title)
    _save(fig, Path(path))


def heatmap_matrix(matrix: pd.DataFrame, title: str, xlabel: str, ylabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use a consistent blue palette; avoid interpolation artifacts
    im = ax.imshow(matrix.values, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    _style(ax, title, xlabel, ylabel)
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    _save(fig, Path(path))


def bar(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=BASE_BLUE, zorder=3)
    _style(ax, title, xlabel, ylabel, rotate_x=True)
    _save(fig, Path(path))


def scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
    invert_y: bool = False,
    add_trendline: bool = True,
    add_binned_avg: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    mask = xs.notna() & ys.notna()
    xs = xs[mask]
    ys = ys[mask]
    ax.scatter(xs, ys, s=12, alpha=0.6, color=BASE_BLUE, zorder=3)
    # Trendline (OLS)
    if add_trendline and len(xs) >= 3:
        try:
            coeffs = np.polyfit(xs, ys, 1)
            xx = np.linspace(xs.min(), xs.max(), 100)
            yy = coeffs[0] * xx + coeffs[1]
            ax.plot(xx, yy, color=BLUE_DARK, linewidth=1.6, label="Trendline", zorder=2)
        except Exception:
            pass
    # Binned rolling average
    if add_binned_avg and len(xs) >= 20:
        try:
            order = np.argsort(xs.values)
            xs_sorted = xs.values[order]
            ys_sorted = ys.values[order]
            win = max(5, len(xs_sorted) // 20)
            yy = pd.Series(ys_sorted).rolling(win, min_periods=max(3, win // 2)).mean()
            ax.plot(xs_sorted, yy, color=GREY_MED, linewidth=1.6, label="Rolling avg", zorder=2)
        except Exception:
            pass
    _style(ax, title, xlabel, ylabel, invert_y=invert_y)
    if add_trendline or add_binned_avg:
        ax.legend(loc="best", fontsize=8)
    _save(fig, Path(path))


def scatter_start_locations_with_labels(
    lon: pd.Series,
    lat: pd.Series,
    clusters_labeled: pd.DataFrame,
    *,
    precision: float = 0.003,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(lon, lat, s=6, alpha=0.5, color=BASE_BLUE)

    def clus_to_latlon(c: str) -> tuple[float, float]:
        a, b = str(c).split(":")
        # cluster stored as lat_c:lon_c
        lat_c = int(a) * precision
        lon_c = int(b) * precision
        return lon_c, lat_c  # x=lon, y=lat

    ann = clusters_labeled.head(30)
    for _, r in ann.iterrows():
        try:
            x, y = clus_to_latlon(str(r["cluster"]))
            lbl = str(r["label"]) if pd.notna(r["label"]) else ""
            if lbl:
                ax.annotate(lbl, (x, y), fontsize=8, xytext=(4, 2), textcoords="offset points", color=BLUE_DARK)
        except Exception:
            continue
    _style(ax, "Start Locations", "Longitude", "Latitude")
    _save(fig, Path(path))


def stack_area(
    x: list[str],
    series: dict[str, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(series.keys())
    data = [series[k].reindex(x).fillna(0).values for k in keys]
    ax.stackplot(x, *data, labels=keys)
    _style(ax, title, xlabel, ylabel)
    ax.legend(loc="upper left")
    _save(fig, Path(path))


def calendar_heatmap(
    matrix: pd.DataFrame, year: int, path: str, vmin: float | None = None, vmax: float | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks([])
    ax.set_title(f"{year} Training Calendar (miles per day)")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Miles")
    ax.grid(False)
    _save(fig, Path(path))


def line_daily_rollup(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    path: str,
    add_trendline: bool = True,
    long_roll_window: int = 30,
) -> None:
    if value_col not in df:
        return
    fig, ax = plt.subplots(figsize=(9, 3.5))
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])  # assume local or naive ok
    y = pd.to_numeric(d[value_col], errors="coerce")
    # Treat zeros as missing for trendlines; start at first non-zero
    nonzero = (y > 0).fillna(False)
    if nonzero.any():
        first_idx = nonzero.idxmax()
        d = d.loc[first_idx:]
        y = y.loc[first_idx:]
    ax.plot(d["date"], y, linewidth=1.4, color=BASE_BLUE, label="daily", zorder=3)
    if long_roll_window and long_roll_window > 1:
        roll = y.rolling(long_roll_window, min_periods=max(5, long_roll_window // 3)).mean()
        ax.plot(d["date"], roll, color=GREY_MED, linewidth=1.6, label=f"{long_roll_window}-day avg", zorder=2)
    if add_trendline and (y[y > 0].notna().sum() >= 10):
        try:
            x = (d["date"] - d["date"].min()).dt.days.astype(float)
            ok = y.notna() & (y > 0)
            coeffs = np.polyfit(x[ok], y[ok], 1)
            xx = np.linspace(float(x.min()), float(x.max()), 100)
            yy = coeffs[0] * xx + coeffs[1]
            ax.plot(
                d["date"].min() + pd.to_timedelta(xx, unit="D"),
                yy,
                color=BLUE_DARK,
                linewidth=1.6,
                label="Trend",
                zorder=2,
            )
        except Exception:
            pass
    _style(ax, title, "Date", ylabel)
    ax.legend(loc="best", fontsize=8)
    _save(fig, Path(path))


def bar_counts(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: str) -> None:
    if df.empty or label_col not in df or value_col not in df:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[label_col].astype(str), df[value_col].astype(float), color="#4c72b0")
    _style(ax, title, label_col, "Count", rotate_x=True)
    _save(fig, Path(path))
