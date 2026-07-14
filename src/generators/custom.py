"""
Custom Generator - Advanced Pattern Learning from User Data
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import warnings
from logger import get_logger
from utils.exceptions import GenerationError

logger = get_logger()
warnings.filterwarnings('ignore')

class CustomGenerator:
    """Advanced synthetic data generator from user-provided sample"""
    
    def __init__(self, sample_data: pd.DataFrame):
        try:
            if sample_data is None:
                raise GenerationError("Sample data is None")
            
            if len(sample_data) < 2:
                logger.warning(f"Sample has only {len(sample_data)} rows. Duplicating...")
                self.sample = pd.concat([sample_data, sample_data], ignore_index=True)
            else:
                self.sample = sample_data
            
            self.models = {}
            self.column_types = {}
            self.correlations = None
            self._validate_sample()
            self._build_models()
            self._calculate_correlations()
            logger.info(f"CustomGenerator initialized with {len(self.sample)} rows")
            
        except Exception as e:
            logger.error(f"Failed to initialize CustomGenerator: {str(e)}")
            raise GenerationError(f"CustomGenerator initialization failed: {str(e)}")
    
    def _validate_sample(self):
        """Validate sample data"""
        try:
            if len(self.sample) == 0:
                raise GenerationError("Sample is empty")
            
            if len(self.sample.columns) == 0:
                raise GenerationError("Sample has no columns")
            
            for col in self.sample.columns:
                if self.sample[col].isnull().all():
                    logger.warning(f"Column '{col}' is all null values")
            
            logger.debug("Sample validation passed")
            
        except Exception as e:
            logger.error(f"Sample validation failed: {str(e)}")
            raise GenerationError(f"Sample validation failed: {str(e)}")
    
    def _build_models(self):
        """Build generation models with error handling"""
        try:
            for col in self.sample.columns:
                try:
                    col_type = self._detect_column_type(self.sample[col])
                    self.column_types[col] = col_type
                    
                    if col_type == 'numeric':
                        self.models[col] = self._build_numeric_model(self.sample[col])
                    elif col_type == 'datetime':
                        self.models[col] = self._build_datetime_model(self.sample[col])
                    elif col_type == 'categorical':
                        self.models[col] = self._build_categorical_model(self.sample[col])
                    else:
                        self.models[col] = {'type': 'constant', 'value': 'Unknown'}
                    
                    logger.debug(f"Built model for column '{col}' with type '{col_type}'")
                    
                except Exception as e:
                    logger.warning(f"Failed to build model for column '{col}': {str(e)}")
                    self.models[col] = {'type': 'constant', 'value': 'Unknown'}
            
            logger.info(f"Built {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Model building failed: {str(e)}")
            raise GenerationError(f"Model building failed: {str(e)}")
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect column type with error handling"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                return 'numeric'
            elif pd.api.types.is_datetime64_dtype(series):
                return 'datetime'
            elif pd.api.types.is_bool_dtype(series):
                return 'categorical'
            else:
                unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
                if unique_ratio < 0.5 and series.nunique() < 20:
                    return 'categorical'
                return 'text'
        except Exception as e:
            logger.warning(f"Column type detection failed: {str(e)}")
            return 'text'
    
    def _build_numeric_model(self, series: pd.Series) -> dict:
        """Build numeric model with error handling"""
        try:
            values = series.dropna().values
            
            if len(values) < 3:
                return {
                    'type': 'numeric',
                    'method': 'statistical',
                    'mean': np.mean(values) if len(values) > 0 else 0,
                    'std': np.std(values) if len(values) > 1 else 1,
                    'min': np.min(values) if len(values) > 0 else 0,
                    'max': np.max(values) if len(values) > 0 else 1
                }
            
            try:
                kde = KernelDensity(kernel='gaussian', bandwidth='scott')
                kde.fit(values.reshape(-1, 1))
                
                return {
                    'type': 'numeric',
                    'method': 'kde',
                    'kde': kde,
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 1
                }
            except Exception as e:
                logger.warning(f"KDE failed, using statistical method: {str(e)}")
                return {
                    'type': 'numeric',
                    'method': 'statistical',
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 1,
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
        except Exception as e:
            logger.error(f"Numeric model building failed: {str(e)}")
            return {
                'type': 'numeric',
                'method': 'statistical',
                'mean': 0,
                'std': 1,
                'min': 0,
                'max': 1
            }
    
    def _build_categorical_model(self, series: pd.Series) -> dict:
        """Build categorical model with error handling"""
        try:
            value_counts = series.value_counts(normalize=True)
            
            if len(value_counts) == 0:
                return {
                    'type': 'categorical',
                    'values': ['Unknown'],
                    'probabilities': [1.0]
                }
            
            return {
                'type': 'categorical',
                'values': value_counts.index.tolist(),
                'probabilities': value_counts.values.tolist()
            }
            
        except Exception as e:
            logger.error(f"Categorical model building failed: {str(e)}")
            return {
                'type': 'categorical',
                'values': ['Unknown'],
                'probabilities': [1.0]
            }
    
    def _build_datetime_model(self, series: pd.Series) -> dict:
        """Build datetime model with error handling"""
        try:
            timestamps = series.dropna().astype('int64') // 10**9
            
            if len(timestamps) == 0:
                return {
                    'type': 'datetime',
                    'min': pd.Timestamp.now(),
                    'max': pd.Timestamp.now(),
                    'mean_ts': 0,
                    'std_ts': 1
                }
            
            return {
                'type': 'datetime',
                'min': series.min(),
                'max': series.max(),
                'mean_ts': timestamps.mean(),
                'std_ts': timestamps.std() if len(timestamps) > 1 else 1
            }
            
        except Exception as e:
            logger.error(f"Datetime model building failed: {str(e)}")
            return {
                'type': 'datetime',
                'min': pd.Timestamp.now(),
                'max': pd.Timestamp.now(),
                'mean_ts': 0,
                'std_ts': 1
            }
    
    def _calculate_correlations(self):
        """Calculate correlations with error handling"""
        try:
            numeric_cols = [col for col, typ in self.column_types.items() 
                           if typ == 'numeric' and col in self.sample.columns]
            
            if len(numeric_cols) > 1:
                self.correlations = self.sample[numeric_cols].corr()
                logger.debug(f"Calculated correlations for {len(numeric_cols)} numeric columns")
            else:
                self.correlations = None
                
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {str(e)}")
            self.correlations = None
    
    # ========== FIXED METHOD ==========
    def generate(self, n: int, preserve_correlations: bool = True) -> pd.DataFrame:
        """Generate synthetic data with error handling"""
        try:
            if n < 1:
                raise GenerationError(f"Number of records must be at least 1: {n}")
            
            if n > 100000:
                logger.warning(f"Large generation requested: {n} records")
            
            data = {}
            
            for col, model in self.models.items():
                try:
                    data[col] = self._generate_column(model, n)
                except Exception as e:
                    logger.warning(f"Failed to generate column '{col}': {str(e)}")
                    data[col] = np.array(['Unknown'] * n)
            
            df = pd.DataFrame(data)
            
            if preserve_correlations and self.correlations is not None:
                try:
                    df = self._apply_correlations(df)
                except Exception as e:
                    logger.warning(f"Correlation application failed: {str(e)}")
            
            try:
                df = self._apply_constraints(df)
            except Exception as e:
                logger.warning(f"Constraint application failed: {str(e)}")
            
            logger.info(f"Generated {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise GenerationError(f"Generation failed: {str(e)}")
    
    def _generate_column(self, model: dict, n: int) -> np.ndarray:
        """Generate values for a column with error handling"""
        try:
            if model['type'] == 'numeric':
                if model.get('method') == 'kde' and 'kde' in model:
                    try:
                        samples = model['kde'].sample(n).flatten()
                        return np.clip(samples, model['min'], model['max'])
                    except:
                        pass
                return np.random.normal(model['mean'], model['std'], n)
                
            elif model['type'] == 'datetime':
                timestamps = np.random.normal(model['mean_ts'], model['std_ts'], n)
                timestamps = np.clip(timestamps, 
                                    model['min'].timestamp(), 
                                    model['max'].timestamp())
                return pd.to_datetime(timestamps, unit='s')
                
            elif model['type'] == 'categorical':
                if len(model['values']) > 1:
                    return np.random.choice(model['values'], n, p=model['probabilities'])
                else:
                    return np.array([model['values'][0]] * n)
            
            else:
                return np.array(['Unknown'] * n)
                
        except Exception as e:
            logger.warning(f"Column generation failed: {str(e)}")
            return np.array(['Unknown'] * n)
    
    def _apply_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply correlations with error handling"""
        try:
            numeric_cols = [col for col in self.column_types 
                           if self.column_types[col] == 'numeric' and col in df.columns]
            
            if len(numeric_cols) < 2:
                return df
            
            gen_data = df[numeric_cols].values
            scaler = StandardScaler()
            gen_scaled = scaler.fit_transform(gen_data)
            L = np.linalg.cholesky(self.correlations.values + np.eye(len(numeric_cols)) * 0.001)
            correlated = gen_scaled @ L.T
            correlated = scaler.inverse_transform(correlated)
            
            for i, col in enumerate(numeric_cols):
                df[col] = correlated[:, i]
                
            return df
            
        except Exception as e:
            logger.warning(f"Correlation application failed: {str(e)}")
            return df
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply constraints with error handling"""
        try:
            for col in df.columns:
                if col in self.models:
                    model = self.models[col]
                    if model['type'] == 'numeric' and 'min' in model and 'max' in model:
                        df[col] = np.clip(df[col], model['min'], model['max'])
            return df
            
        except Exception as e:
            logger.warning(f"Constraint application failed: {str(e)}")
            return df