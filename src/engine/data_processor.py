"""
Data Processor - Cleaning & Preprocessing
"""

import pandas as pd
import numpy as np

class DataProcessor:
    """Process and clean data"""
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe"""
        
        df = data.copy()
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill nulls for numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill nulls for categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if len(df[col]) > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        return df
    
    def process_sample(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Process sample data"""
        
        df = sample.copy()
        
        # Convert date columns
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = df[col].astype('datetime64[ns]')
        
        return df