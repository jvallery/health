from __future__ import annotations


def to_meters(value: str | float | None, unit: str | None) -> float | None:
    if value is None or unit is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    u = unit.lower()
    if u in {"m", "meter", "meters"}:
        return v
    if u in {"km", "kilometer", "kilometers"}:
        return v * 1000.0
    if u in {"mi", "mile", "miles"}:
        return v * 1609.344
    if u in {"ft", "feet"}:
        return v * 0.3048
    return v


def to_kcal(value: str | float | None, unit: str | None) -> float | None:
    if value is None or unit is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    u = unit.lower()
    if u in {"kcal", "kilocalorie", "kilocalories"}:
        return v
    if u in {"cal", "calorie", "calories"}:
        return v / 1000.0
    if u in {"kj", "kilojoule", "kilojoules"}:
        return v * 0.239005736
    return v
