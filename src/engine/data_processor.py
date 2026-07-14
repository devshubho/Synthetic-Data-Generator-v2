"""
Data Processor with Error Handling
"""

import pandas as pd
import numpy as np
from src.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger()

class DataProcessor:
    """Process and clean data with error handling"""
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe with error handling"""
        try:
            if data is None or len(data) == 0:
                logger.warning("Empty data provided for cleaning")
                return data
            
            df = data.copy()
            original_shape = df.shape
            
            # Remove duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                df = df.drop_duplicates()
                logger.info(f"Removed {duplicates} duplicate rows")
            
            # Fill nulls for numeric with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        logger.debug(f"Filled nulls in '{col}' with median: {median_val}")
                except Exception as e:
                    logger.warning(f"Could not fill nulls in '{col}': {str(e)}")
            
            # Fill nulls for categorical with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                try:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col] = df[col].fillna(mode_val)
                        logger.debug(f"Filled nulls in '{col}' with mode: {mode_val}")
                except Exception as e:
                    logger.warning(f"Could not fill nulls in '{col}': {str(e)}")
            
            # Remove columns that are all null
            all_null_cols = [col for col in df.columns if df[col].isnull().all()]
            if all_null_cols:
                df = df.drop(columns=all_null_cols)
                logger.info(f"Removed all-null columns: {all_null_cols}")
            
            logger.info(f"Cleaned data: {original_shape} -> {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise DataValidationError(f"Data cleaning failed: {str(e)}")
    
    def process_sample(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Process sample data with error handling"""
        try:
            if sample is None or len(sample) == 0:
                raise DataValidationError("Sample is empty")
            
            df = sample.copy()
            
            # Convert date columns
            for col in df.columns:
                try:
                    if pd.api.types.is_datetime64_dtype(df[col]):
                        df[col] = df[col].astype('datetime64[ns]')
                except Exception as e:
                    logger.warning(f"Could not process datetime column '{col}': {str(e)}")
            
            # Standardize string columns
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