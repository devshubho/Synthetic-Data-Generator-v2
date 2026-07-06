"""
CSV Export
"""

import pandas as pd
import io

class CSVExporter:
    """Export data to CSV"""
    
    def export(self, df: pd.DataFrame) -> bytes:
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode()