"""
<<<<<<< HEAD
Data Processor with Intelligent Deduplication
=======
Data Processor - Cleaning & Preprocessing
>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
"""

import pandas as pd
import numpy as np
<<<<<<< HEAD
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
            
            # === STEP 3: STANDARDIZE ===
            # Standardize string columns
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                try:
                    df[col] = df[col].astype(str).str.strip()
                    # Replace empty strings with None
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
            
            # Apply deduplication
            df = self.deduplicator.deduplicate(df, data_type)
            
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
    
    def get_dedup_report(self) -> Dict:
        """Get deduplication report"""
        return self.deduplicator.get_dedup_report()
    
    def analyze_duplicates(self, data: pd.DataFrame) -> Dict:
        """Analyze duplicates in data"""
        return self.deduplicator.get_duplicates_info(data)
=======

class DataProcessor:
    """Process and clean data"""
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe"""
        
        df = data.copy()
        integer_cols = [
            col for col in df.columns
            if pd.api.types.is_integer_dtype(df[col])
        ]
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill nulls for numeric with median (preserve integer dtypes)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not df[col].isna().any():
                continue
            median = df[col].median()
            if pd.isna(median):
                median = 0
            if col in integer_cols or pd.api.types.is_integer_dtype(df[col]):
                median = int(round(float(median)))
            df[col] = df[col].fillna(median)
        
        # Restore integer columns after fillna (which may upcast to float)
        for col in integer_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors='coerce')
                    .fillna(0)
                    .round()
                    .astype('int64')
                )
        
        # Fill nulls for categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if len(df[col]) > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        return df

    def preserve_numeric_style(self, df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
        """
        Match generated numerics to the sample style:
        - whole-number sample columns stay integers (no decimals)
        - 6-digit codes (e.g. pincode) stay 6-digit integers
        """
        skip_name_re = (
            'aadhaar', 'aadhar', 'ifsc', 'gstin', 'upi', 'phone', 'mobile',
            'pan', 'msisdn', 'imei', 'email',
        )
        result = df.copy()
        for col in sample.columns:
            if col not in result.columns:
                continue

            name = str(col).strip().lower()
            if any(k in name for k in skip_name_re):
                continue

            # Never cast datetime columns to integers (to_numeric yields epoch ns)
            if (
                pd.api.types.is_datetime64_dtype(sample[col])
                or pd.api.types.is_datetime64_dtype(result[col])
            ):
                continue
            try:
                as_dates = pd.to_datetime(sample[col], errors='coerce')
                if as_dates.notna().mean() > 0.8:
                    continue
            except Exception:
                pass

            sample_nums = pd.to_numeric(sample[col], errors='coerce').dropna()
            if len(sample_nums) == 0:
                continue

            # Skip identity-scale integers float64 cannot uniquely represent
            if sample_nums.abs().max() >= 1e11 or (
                sample_nums.astype(str).str.replace(r'\.0$', '', regex=True).str.len().median() >= 11
            ):
                continue

            is_integer_like = (
                pd.api.types.is_integer_dtype(sample[col])
                or np.allclose(sample_nums.values, np.round(sample_nums.values))
            )
            if not is_integer_like:
                continue

            values = pd.to_numeric(result[col], errors='coerce')
            fill = int(round(float(sample_nums.median())))
            values = values.fillna(fill).values
            values = np.rint(values).astype(np.int64)

            sample_ints = np.rint(sample_nums.values).astype(np.int64)
            if np.all((sample_ints >= 100000) & (sample_ints <= 999999)):
                values = np.clip(values, 100000, 999999)
            else:
                values = np.clip(values, int(sample_ints.min()), int(sample_ints.max()))

            result[col] = values

        return result
    def process_sample(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Process sample data"""
        
        df = sample.copy()
        
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = df[col].astype('datetime64[ns]')
                continue

            # Clean dirty object labels (e.g. Excel 'FALSE ' with trailing spaces)
            if df[col].dtype == object or str(df[col].dtype) == 'string':
                df[col] = df[col].map(self._normalize_cell)
        
        return df

    @staticmethod
    def _normalize_cell(value):
        """Normalize bool-like / messy string cells without changing real numbers."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        if isinstance(value, bool):
            return 'TRUE' if value else 'FALSE'
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return value

        text = str(value).strip()
        lowered = text.lower()
        bool_map = {
            'true': 'TRUE', 'false': 'FALSE',
            'yes': 'TRUE', 'no': 'FALSE',
            'y': 'TRUE', 'n': 'FALSE',
            't': 'TRUE', 'f': 'FALSE',
            '1': 'TRUE', '0': 'FALSE',
        }
        if lowered in bool_map:
            return bool_map[lowered]
        return text
>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
