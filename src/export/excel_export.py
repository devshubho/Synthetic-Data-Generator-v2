"""
Excel Export
"""

import pandas as pd
import io


def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with timezone-aware datetimes converted to naive (Excel requirement)."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if isinstance(series.dtype, pd.DatetimeTZDtype) or getattr(series.dtype, "tz", None) is not None:
            out[col] = series.dt.tz_localize(None)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        # Object columns may hold tz-aware Timestamp / datetime values
        if series.dtype == object:
            sample = series.dropna().head(50)
            if len(sample) == 0:
                continue
            if not all(isinstance(v, (pd.Timestamp,)) or hasattr(v, "tzinfo") for v in sample):
                continue
            if any(getattr(v, "tzinfo", None) is not None for v in sample):
                converted = pd.to_datetime(series, errors="coerce", utc=True)
                out[col] = converted.dt.tz_localize(None)
    return out


class ExcelExporter:
    """Export data to Excel"""

    def export(self, df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        safe_df = make_excel_safe(df)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            safe_df.to_excel(writer, index=False, sheet_name="Data")
        return output.getvalue()
