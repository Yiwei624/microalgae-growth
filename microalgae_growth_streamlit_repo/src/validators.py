from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

REQUIRED: Dict[str, List[str]] = {
    "study": ["title"],
    "organism": ["genus", "species"],
    "experiment": ["exp_code"],
}

RANGES = {
    ("experiment", "temperature_C"): (-10, 60),
    ("experiment", "pH"): (0, 14),
    ("experiment", "light_uE_m2_s"): (0, 5000),
    ("experiment", "gas_co2_percent"): (0, 100),
    ("experiment", "gas_o2_percent"): (0, 100),
    ("experiment", "dissolved_oxygen_mgL"): (0, 30),
}

def validate_df(table: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for col in REQUIRED.get(table, []):
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    for col in REQUIRED.get(table, []):
        if col in df.columns and df[col].isna().all():
            errors.append(f"Required column has all NA: {col}")

    for (t, col), (lo, hi) in RANGES.items():
        if t == table and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                bad = s[(s < lo) | (s > hi)]
                if len(bad) > 0:
                    warnings.append(f"{col}: {len(bad)} values outside [{lo}, {hi}] (soft warning)")

    return errors, warnings
