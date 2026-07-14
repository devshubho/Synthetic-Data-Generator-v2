"""
Data Processor with Intelligent Deduplication
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from logger import get_logger
from utils.exceptions import DataValidationError
from utils.deduplicator import DataDeduplicator

logger = get_logger()

class DataProcessor:
    """Process and clean data with intelligent deduplication"""
    
    def __init__(self):
        self.deduplicator = DataDeduplicator()
    
    def clean(self, data: pd.DataFrame, data_type: str = "unknown") -> pd.DataFrame:
        """Clean dataframe with intelligent deduplication"""
        try:
            if data is None or len(data) == 0:
                logger.warning("Empty data provided for cleaning")
                return data
            
            df = data.copy()
            original_shape = df.shape
            
            # === STEP 1: INTELLIGENT DEDUPLICATION ===
            df = self.deduplicator.deduplicate(df, data_type)
            
            # === STEP 2: FILL NULLS ===
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        logger.debug(f"Filled nulls in '{col}' with median: {median_val}")
                except Exception as e:
                    logger.warning(f"Could not fill nulls in '{col}': {str(e)}")
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                try:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col] = df[col].fillna(mode_val)
                        logger.debug(f"Filled nulls in '{col}' with mode: {mode_val}")
                except Exception as e:
                    logger.warning(f"Could not fill nulls in '{col}': {str(e)}")
            
            # === STEP 3: STANDARDIZE ===
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                try:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace('', None)
                except Exception as e:
                    logger.warning(f"Could not standardize column '{col}': {str(e)}")
            
            # Convert date columns
            for col in df.columns:
                try:
                    if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                        if not pd.api.types.is_datetime64_dtype(df[col]):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert datetime column '{col}': {str(e)}")
            
            logger.info(f"Cleaned data: {original_shape} → {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise DataValidationError(f"Data cleaning failed: {str(e)}")
    
    def process_sample(self, sample: pd.DataFrame, data_type: str = "unknown") -> pd.DataFrame:
        """Process sample data with deduplication"""
        try:
            if sample is None or len(sample) == 0:
                raise DataValidationError("Sample is empty")
            
            df = sample.copy()
            df = self.deduplicator.deduplicate(df, data_type)
            
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                try:
                    df[col] = df[col].astype(str).str.strip()
                except Exception as e:
                    logger.warning(f"Could not process string column '{col}': {str(e)}")
            
            logger.info(f"Processed sample: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Sample processing failed: {str(e)}")
            raise DataValidationError(f"Sample processing failed: {str(e)}")
    
    def get_dedup_report(self) -> Dict[str, Any]:
        """Get deduplication report"""
        return self.deduplicator.get_dedup_report()
    
    def analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicates in data"""
        return self.deduplicator.get_duplicates_info(data)