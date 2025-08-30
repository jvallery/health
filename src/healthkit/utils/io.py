from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet_csv(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(base) + ".parquet", index=False)
    df.to_csv(str(base) + ".csv", index=False)
