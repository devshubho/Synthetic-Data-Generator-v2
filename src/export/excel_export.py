"""
Excel Export
"""

import pandas as pd
import io

class ExcelExporter:
    """Export data to Excel"""
    
    def export(self, df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        return output.getvalue()