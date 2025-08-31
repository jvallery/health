from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .palette import PALETTE, FIG_SIZE, DPI


def _setup():
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
            "grid.color": "#e6e9ef",
            "grid.linewidth": 0.6,
        }
    )


_setup()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def line(
    x: Iterable,
    y: Iterable,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str,
    invert_y: bool = False,
    rolling: int | None = None,
    rolling_color: str | None = None,
    min_periods: int = 1,
    rotate_x: bool = False,
    markers: bool = False,
    y_quantile_clip: tuple[float, float] | None = None,
    rolling_mode: str = "median",  # "median" or "mean"
    grid: bool = True,
    show_raw: bool = True,
    raw_color: str | None = None,
    raw_alpha: float = 0.35,
    raw_linewidth: float = 1.2,
    roll_linewidth: float = 2.2,
    integer_y: bool = False,
) -> None:
    # Build index from x; ensure datetimes are naive for matplotlib
    x_list = list(x)
    idx = pd.Index(x_list, name=xlabel)
    if isinstance(idx, pd.DatetimeIndex):
        try:
            # If tz-aware, convert to naive; if naive, this will raise and fall through
            idx = idx.tz_convert(None)
        except Exception:
            pass
    # Important: avoid label alignment if y is a pandas Series by forcing to array
    y_vals = pd.to_numeric(pd.Series(list(y)), errors="coerce").to_numpy()
    s = pd.Series(y_vals, index=idx)
    # Ensure NaNs break lines
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.grid(grid)
    # Draw raw series (faint), and overlay rolling as the main colored line
    if s.notna().sum() >= 1 and show_raw:
        ax.plot(
            s.index,
            s.values,
            color=raw_color or PALETTE["baseline"],
            linewidth=raw_linewidth,
            alpha=raw_alpha,
            zorder=1,
            marker=None,
        )
    if rolling and rolling > 1:
        if rolling_mode == "mean":
            roll = s.rolling(rolling, min_periods=min_periods).mean()
        else:
            roll = s.rolling(rolling, min_periods=min_periods).median()
        if roll.notna().sum() >= 1:
            ax.plot(
                roll.index,
                roll.values,
                color=rolling_color or PALETTE["baseline"],
                linewidth=roll_linewidth,
                linestyle="-",
                zorder=3,
            )
    elif s.notna().sum() >= 1:
        # No rolling requested; draw the raw series as the primary line
        ax.plot(
            s.index,
            s.values,
            color=color,
            linewidth=roll_linewidth,
            zorder=3,
            marker="o" if markers else None,
            markersize=5 if markers else None,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if invert_y:
        ax.invert_yaxis()
    if rotate_x:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")
        # Downsample monthly labels like YYYY-MM to reduce clutter
        try:
            labels = list(idx.astype(str)) if not isinstance(idx, pd.DatetimeIndex) else [t.strftime('%Y-%m') for t in idx]
            if len(labels) > 24 and all(len(s) >= 7 and s[4] == '-' for s in labels):
                step = max(1, len(labels)//12 * 2)
                sel = list(range(0, len(labels), step))
                ax.set_xticks([idx[i] for i in sel])
        except Exception:
            pass
    if y_quantile_clip is not None and s.notna().sum() >= 5:
        lo, hi = y_quantile_clip
        ql = float(s.quantile(lo))
        qh = float(s.quantile(hi))
        if ql == ql and qh == qh and qh > ql:
            pad = 0.05 * (qh - ql)
            ax.set_ylim(ql - pad, qh + pad)
    if integer_y:
        try:
            import matplotlib.ticker as mticker
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        except Exception:
            pass
    if (s.notna().sum() == 0) and not (rolling and roll.notna().sum()):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=PALETTE["baseline"])
    _save(fig, path)


def bar(
    x: Iterable,
    y: Iterable,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    vals = pd.to_numeric(pd.Series(list(y)), errors="coerce")
    if vals.notna().sum() == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=PALETTE["baseline"])
    else:
        ax.bar(list(x), vals.values, color=color, zorder=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def scatter(
    x: Iterable,
    y: Iterable,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str,
    alpha: float = 0.7,
    add_linreg: bool = False,
) -> None:
    x = pd.to_numeric(pd.Series(list(x)), errors="coerce")
    y = pd.to_numeric(pd.Series(list(y)), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if len(x) == 0 or len(y) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=PALETTE["baseline"])
    else:
        ax.scatter(x, y, c=color, alpha=alpha, edgecolors="none")
    if add_linreg and len(x) >= 3:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(float(x.min()), float(x.max()), 100)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, color=PALETTE["baseline"], linestyle="--", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def _tint(hexcolor: str, factor: float) -> str:
    hexcolor = hexcolor.lstrip("#")
    r, g, b = tuple(int(hexcolor[i : i + 2], 16) for i in (0, 2, 4))
    r = int(min(255, r + (255 - r) * factor))
    g = int(min(255, g + (255 - g) * factor))
    b = int(min(255, b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def donut(labels: list[str], sizes: list[float], *, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    base = PALETTE["hr"]
    colors = [_tint(base, f) for f in [0.0, 0.15, 0.3, 0.45, 0.6]][: len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.25),
        textprops=dict(color="#2b3e50"),
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.set_title(title)
    _save(fig, path)


def stack_area(
    x: Iterable,
    series: Mapping[str, Iterable[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    keys = list(series.keys())
    cols = [PALETTE["volume"], PALETTE["counts"], PALETTE["pace"], PALETTE["hr"], PALETTE["elevation"]]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ys = np.row_stack([pd.to_numeric(pd.Series(list(series[k])), errors="coerce").fillna(0).values for k in keys])
    ax.stackplot(list(x), ys, labels=keys, colors=cols[: len(keys)], alpha=0.9)
    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def calendar_heatmap(matrix: np.ndarray, year: int, path: Path, *, vmin: float | None, vmax: float | None) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="Blues")
    ax.set_title(f"Calendar Heatmap {year}")
    fig.colorbar(im, ax=ax, label="Miles")
    _save(fig, path)
