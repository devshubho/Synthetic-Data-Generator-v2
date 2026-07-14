"""
Data Validator with Error Handling
"""

import pandas as pd
import numpy as np
from src.utils.exceptions import DataValidationError, SampleError
from src.logger import get_logger

logger = get_logger()

class DataValidator:
    """Validate data with comprehensive error handling"""
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate dataframe"""
        try:
            if data is None:
                raise DataValidationError("Data is None")
            
            if len(data) == 0:
                raise DataValidationError("Data is empty")
            
            if len(data.columns) == 0:
                raise DataValidationError("No columns found in data")
            
            # Check for excessive nulls
            total_cells = data.shape[0] * data.shape[1]
            if total_cells == 0:
                raise DataValidationError("Data has zero cells")
                
            null_count = data.isnull().sum().sum()
            null_percentage = null_count / total_cells
            
            if null_percentage > 0.9:
                raise DataValidationError(
                    f"Too many null values: {null_percentage:.1%}. "
                    f"Please clean your data first."
                )
            
            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if np.isinf(data[col]).any().any():
                    raise DataValidationError(
                        f"Column '{col}' contains infinite values"
                    )
            
            logger.info(f"Data validation passed: {len(data)} rows, {len(data.columns)} columns")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected validation error: {str(e)}")
            raise DataValidationError(f"Validation failed: {str(e)}")
    
    def validate_sample(self, sample: pd.DataFrame) -> bool:
        """Validate sample data"""
        try:
            if sample is None:
                raise SampleError("Sample data is None")
            
            if len(sample) < 2:
                raise SampleError(
                    f"Sample needs at least 2 records. Current: {len(sample)}. "
                    f"Please upload more data."
                )
            
            if len(sample.columns) == 0:
                raise SampleError("Sample has no columns")
            
            # Check for all null columns
            for col in sample.columns:
                if sample[col].isnull().all():
                    logger.warning(f"Column '{col}' is all null values")
            
            logger.info(f"Sample validation passed: {len(sample)} rows, {len(sample.columns)} columns")
            return True
            
        except SampleError:
            raise
        except Exception as e:
            logger.error(f"Unexpected sample validation error: {str(e)}")
            raise SampleError(f"Sample validation failed: {str(e)}")