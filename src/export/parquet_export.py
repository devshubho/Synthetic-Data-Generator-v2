"""
Parquet Export
"""

import pandas as pd
import io

class ParquetExporter:
    """Export data to Parquet"""
    
    def export(self, df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        df.to_parquet(output, index=False)
        return output.getvalue()