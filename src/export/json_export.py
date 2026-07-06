"""
JSON Export
"""

import pandas as pd

class JSONExporter:
    """Export data to JSON"""
    
    def export(self, df: pd.DataFrame) -> bytes:
        return df.to_json(orient='records', indent=2).encode()