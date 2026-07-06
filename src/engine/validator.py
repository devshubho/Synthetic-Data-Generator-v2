"""
Data Validator
"""

import pandas as pd

class DataValidator:
    """Validate data for generation"""
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate dataframe"""
        
        if data is None or len(data) == 0:
            raise ValueError("Data is empty")
        
        if len(data.columns) == 0:
            raise ValueError("No columns found")
        
        # Check for excessive nulls
        null_percentage = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if null_percentage > 0.9:
            raise ValueError(f"Too many null values: {null_percentage:.1%}")
        
        return True
    
    def validate_sample(self, sample: pd.DataFrame) -> bool:
        """Validate sample data"""
        
        if len(sample) < 2:
            raise ValueError("Sample needs at least 2 records")
        
        if len(sample.columns) == 0:
            raise ValueError("Sample has no columns")
        
        return True