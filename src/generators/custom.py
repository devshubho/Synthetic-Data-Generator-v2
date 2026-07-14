"""
Custom Generator - Advanced Pattern Learning from User Data
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CustomGenerator:
    """
    Advanced synthetic data generator from user-provided sample
    Uses multiple ML techniques for accurate pattern preservation
    """
    
    def __init__(self, sample_data: pd.DataFrame):
        self.sample = sample_data
        self.models = {}
        self.column_types = {}
        self.correlations = None
        self._validate_sample()
        self._build_models()
        self._calculate_correlations()
    
    def _validate_sample(self):
        """Validate sample data"""
        if len(self.sample) < 2:
            raise ValueError("Sample must have at least 2 rows")
        
        if len(self.sample.columns) == 0:
            raise ValueError("Sample must have at least 1 column")
    
    def _build_models(self):
        """Build generation models from sample"""
        
        for col in self.sample.columns:
            col_type = self._detect_column_type(self.sample[col])
            self.column_types[col] = col_type
            
            if col_type == 'numeric':
                self.models[col] = self._build_numeric_model(self.sample[col])
            elif col_type == 'datetime':
                self.models[col] = self._build_datetime_model(self.sample[col])
            elif col_type == 'categorical':
                self.models[col] = self._build_categorical_model(self.sample[col])
            elif col_type == 'text':
                self.models[col] = self._build_text_model(self.sample[col])
            else:
                self.models[col] = {'type': 'constant', 'value': 'Unknown'}
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect column data type"""
        
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        elif pd.api.types.is_datetime64_dtype(series):
            return 'datetime'
        elif pd.api.types.is_bool_dtype(series):
            return 'categorical'
        else:
            # Check if categorical (few unique values)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.5 and series.nunique() < 20:
                return 'categorical'
            else:
                return 'text'
    
    def _build_numeric_model(self, series: pd.Series) -> dict:
        """Build numeric column model"""
        
        values = series.dropna().values
        
        if len(values) < 5:
            # Small sample - use simple statistics
            return {
                'type': 'numeric',
                'method': 'statistical',
                'mean': np.mean(values),
                'std': np.std(values) or 1,
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q50': np.percentile(values, 50),
                'q75': np.percentile(values, 75),
                'skew': stats.skew(values) if len(values) > 2 else 0,
                'kurtosis': stats.kurtosis(values) if len(values) > 3 else 0
            }
        
        # Try KDE for larger samples
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
                'std': np.std(values),
                'q25': np.percentile(values, 25),
                'q50': np.percentile(values, 50),
                'q75': np.percentile(values, 75),
                'skew': stats.skew(values),
                'kurtosis': stats.kurtosis(values),
                'outlier_lower': np.percentile(values, 25) - 1.5 * (np.percentile(values, 75) - np.percentile(values, 25)),
                'outlier_upper': np.percentile(values, 75) + 1.5 * (np.percentile(values, 75) - np.percentile(values, 25))
            }
        except:
            return {
                'type': 'numeric',
                'method': 'statistical',
                'mean': np.mean(values),
                'std': np.std(values) or 1,
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q50': np.percentile(values, 50),
                'q75': np.percentile(values, 75)
            }
    
    def _build_datetime_model(self, series: pd.Series) -> dict:
        """Build datetime column model"""
        
        timestamps = series.dropna().astype('int64') // 10**9
        
        if len(timestamps) < 3:
            # Small sample
            return {
                'type': 'datetime',
                'min': series.min(),
                'max': series.max(),
                'mean_ts': timestamps.mean(),
                'std_ts': timestamps.std() or 1
            }
        
        return {
            'type': 'datetime',
            'min': series.min(),
            'max': series.max(),
            'mean_ts': timestamps.mean(),
            'std_ts': timestamps.std(),
            'date_range_days': (series.max() - series.min()).days
        }
    
    def _build_categorical_model(self, series: pd.Series) -> dict:
        """Build categorical column model"""
        
        value_counts = series.value_counts(normalize=True)
        
        # Calculate entropy for diversity
        probs = value_counts.values
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 0 else 1
        
        return {
            'type': 'categorical',
            'values': value_counts.index.tolist(),
            'probabilities': value_counts.values.tolist(),
            'unique_count': len(value_counts),
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy if max_entropy > 0 else 0,
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_common_freq': value_counts.values[0] if len(value_counts) > 0 else 0
        }
    
    def _build_text_model(self, series: pd.Series) -> dict:
        """Build text column model"""
        
        # Text statistics
        text_lengths = series.dropna().str.len()
        
        if len(text_lengths) > 0:
            common_words = series.str.split().explode().value_counts().head(10)
            
            return {
                'type': 'text',
                'min_length': text_lengths.min(),
                'max_length': text_lengths.max(),
                'mean_length': text_lengths.mean(),
                'std_length': text_lengths.std() or 1,
                'common_words': common_words.to_dict() if len(common_words) > 0 else {},
                'unique_ratio': series.nunique() / len(series) if len(series) > 0 else 0
            }
        else:
            return {'type': 'constant', 'value': 'Unknown'}
    
    def _calculate_correlations(self):
        """Calculate correlations between numeric columns"""
        
        numeric_cols = [col for col, typ in self.column_types.items() 
                       if typ == 'numeric']
        
        if len(numeric_cols) > 1:
            self.correlations = self.sample[numeric_cols].corr()
        else:
            self.correlations = None
    
    def generate(self, n: int, preserve_correlations: bool = True) -> pd.DataFrame:
        """Generate synthetic data"""
        
        if n < 1:
            raise ValueError("Number of records must be at least 1")
        
        data = {}
        
        for col, model in self.models.items():
            data[col] = self._generate_column(model, n)
        
        df = pd.DataFrame(data)
        
        # Apply correlations
        if preserve_correlations and self.correlations is not None:
            df = self._apply_correlations(df)
        
        # Apply constraints
        df = self._apply_constraints(df)
        
        return df
    
    def _generate_column(self, model: dict, n: int) -> np.ndarray:
        """Generate values for a column"""
        
        model_type = model.get('type', 'constant')
        
        if model_type == 'numeric':
            if model.get('method') == 'kde' and 'kde' in model:
                try:
                    samples = model['kde'].sample(n).flatten()
                    # Clip to realistic range
                    samples = np.clip(samples, model['min'] - 2*model['std'], 
                                      model['max'] + 2*model['std'])
                    return samples
                except:
                    pass
            
            # Fallback to statistical method
            samples = np.random.normal(model['mean'], model['std'], n)
            return np.clip(samples, model['min'], model['max'])
        
        elif model_type == 'datetime':
            timestamps = np.random.normal(model['mean_ts'], model['std_ts'], n)
            timestamps = np.clip(timestamps, 
                                model['min'].timestamp(), 
                                model['max'].timestamp())
            return pd.to_datetime(timestamps, unit='s')
        
        elif model_type == 'categorical':
            if len(model['values']) > 1:
                return np.random.choice(model['values'], n, p=model['probabilities'])
            else:
                return np.array([model['values'][0]] * n)
        
        elif model_type == 'text':
            return self._generate_text(model, n)
        
        else:
            return np.array(['Unknown'] * n)
    
    def _generate_text(self, model: dict, n: int) -> np.ndarray:
        """Generate realistic text"""
        
        texts = []
        for _ in range(n):
            # Generate length
            length = int(np.random.normal(model['mean_length'], model['std_length']))
            length = max(1, min(length, model['max_length']))
            
            # Generate text
            if model.get('common_words'):
                words = list(model['common_words'].keys())
                if words:
                    num_words = max(1, length // 5)
                    selected_words = np.random.choice(words, num_words)
                    text = ' '.join(selected_words)[:length]
                else:
                    text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), length))
            else:
                text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), length))
            
            texts.append(text)
        
        return np.array(texts)
    
    def _apply_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply correlations from original data"""
        
        numeric_cols = [col for col, typ in self.column_types.items() 
                       if typ == 'numeric' and col in df.columns]
        
        if len(numeric_cols) > 1:
            try:
                # Get generated numeric data
                gen_data = df[numeric_cols].values
                
                # Standardize
                scaler = StandardScaler()
                gen_scaled = scaler.fit_transform(gen_data)
                
                # Apply Cholesky decomposition
                corr_matrix = self.correlations.values
                from scipy.linalg import cholesky
                
                # Ensure positive definite
                L = np.linalg.cholesky(corr_matrix + np.eye(len(numeric_cols)) * 0.001)
                
                # Transform
                correlated = gen_scaled @ L.T
                
                # Scale back
                correlated = scaler.inverse_transform(correlated)
                
                # Replace values
                for i, col in enumerate(numeric_cols):
                    df[col] = correlated[:, i]
                    
            except Exception as e:
                pass
        
        return df
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply constraints from original data"""
        
        for col in df.columns:
            if col in self.column_types:
                col_type = self.column_types[col]
                
                if col_type == 'numeric' and col in self.models:
                    model = self.models[col]
                    if 'min' in model and 'max' in model:
                        df[col] = np.clip(df[col], model['min'], model['max'])
                
                elif col_type == 'datetime':
                    model = self.models[col]
                    if 'min' in model and 'max' in model:
                        df[col] = np.clip(df[col], model['min'], model['max'])
        
        return df
    
    def get_data_profile(self) -> dict:
        """Get comprehensive data profile"""
        
        profile = {
            'num_rows': len(self.sample),
            'num_columns': len(self.sample.columns),
            'columns': {},
            'summary': {
                'numeric': 0,
                'categorical': 0,
                'datetime': 0,
                'text': 0
            },
            'quality': {
                'completeness': 1 - self.sample.isnull().sum().sum() / 
                               (self.sample.shape[0] * self.sample.shape[1]),
                'duplicate_rows': self.sample.duplicated().sum()
            }
        }
        
        for col, typ in self.column_types.items():
            profile['summary'][typ] = profile['summary'].get(typ, 0) + 1
            profile['columns'][col] = {
                'type': typ,
                'unique_values': self.sample[col].nunique(),
                'null_count': self.sample[col].isnull().sum()
            }
        
        return profile