from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def parse_ecg_csv(path: str) -> tuple[dict, pd.DataFrame | None]:
    """Robust ECG CSV summary parser for Apple Health variants."""
    p = Path(path)
    # Load as CSV if possible
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    header = df.iloc[0].to_dict() if not df.empty else {}

    # Also scan the first ~40 lines for key:value or comma-separated pairs
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [next(f, "").strip() for _ in range(40)]
    except Exception:
        lines = []

    def find(keys: list[str]) -> str | None:
        # 1) header dict
        for k in keys:
            v = header.get(k)
            if v not in (None, ""):
                return str(v)
        # 2) lines key: value
        for ln in lines:
            for k in keys:
                if ln.lower().startswith(k.lower() + ":"):
                    return ln.split(":", 1)[1].strip()
        # 3) lines "key, value"
        for ln in lines:
            parts = [s.strip() for s in re.split(r"[,;]", ln)]
            if len(parts) >= 2 and any(k.lower() in ln.lower() for k in keys):
                return parts[1]
        # 4) df columns
        for k in keys:
            if k in df.columns:
                v = df[k].iloc[0]
                if pd.notna(v):
                    return str(v)
        return None

    start_raw = find(["Start Date", "start_time", "Date", "Start"])
    start = pd.to_datetime(start_raw, utc=True, errors="coerce")
    sampling = find(["Sampling Frequency", "sampling_hz", "SamplingFrequency"]) or ""
    mean_hr = find(["Heart Rate", "mean_hr_bpm", "AverageHeartRate"]) or ""
    classification = find(["Classification", "Rhythm", "Diagnosis"]) or ""
    device = find(["Device", "device"]) or ""
    hw = None
    m = re.search(r"hardware:([A-Za-z0-9,]+)", device, re.IGNORECASE)
    if m:
        hw = m.group(1)

    def _to_float_safe(val):
        try:
            return float(re.sub(r"[^0-9.]+", "", str(val)))
        except Exception:
            return None

    out = {
        "ecg_id": p.stem,
        "start_utc": start if pd.notna(start) else None,
        "duration_sec": float(header.get("Duration", 30.0)) if header else 30.0,
        "mean_hr_bpm": _to_float_safe(mean_hr),
        "classification": classification or None,
        "symptoms": find(["Symptoms"]),
        "sampling_hz": float(re.sub(r"[^0-9.]+", "", sampling)) if sampling else None,
        "path": p.name,
        "device": device or None,
        "hardware": hw,
    }
    return out, None
