from __future__ import annotations
from io import BytesIO
from typing import Dict, List, Tuple
import pandas as pd

def read_excel_sheets(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(BytesIO(file_bytes))
    out: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name)
        df.columns = [str(c).strip() for c in df.columns]
        out[name.strip().lower()] = df
    return out

def read_csv_files(files: List[Tuple[str, bytes]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for fname, b in files:
        # Support both Linux/URL paths and Windows paths safely
        base_name = fname.replace("\\", "/").split("/")[-1]
        table = base_name.rsplit(".", 1)[0].strip().lower()
        df = pd.read_csv(BytesIO(b))
        df.columns = [str(c).strip() for c in df.columns]
        out[table] = df
    return out
