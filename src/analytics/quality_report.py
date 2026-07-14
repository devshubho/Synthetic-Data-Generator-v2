"""
Quality Report - Comprehensive Analysis with Advanced Features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from logger import get_logger
import io
import json
import zipfile

logger = get_logger()


class QualityReporter:
    """Generate comprehensive quality reports"""
    
    def generate_report(self, data: pd.DataFrame) -> dict:
        """Generate full quality report"""
        return {
            'overall_score': self._overall_score(data),
            'completeness': self._completeness(data),
            'uniqueness': self._uniqueness(data),
            'diversity': self._diversity(data),
            'privacy_score': self._privacy_score(data),
            'statistics': self._statistics(data),
            'advanced_metrics': self.calculate_advanced_metrics(data)
        }
    
    def _overall_score(self, data: pd.DataFrame) -> float:
        scores = [self._completeness(data), self._uniqueness(data), self._diversity(data)]
        return round(sum(scores) / len(scores), 3)
    
    def _completeness(self, data: pd.DataFrame) -> float:
        total = data.shape[0] * data.shape[1]
        if total == 0:
            return 0
        nulls = data.isnull().sum().sum()
        return 1 - (nulls / total)
    
    def _uniqueness(self, data: pd.DataFrame) -> float:
        if len(data) == 0:
            return 0
        unique_ratios = [data[col].nunique() / len(data) for col in data.columns]
        return np.mean(unique_ratios) if unique_ratios else 0
    
    def _diversity(self, data: pd.DataFrame) -> float:
        scores = []
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                mean = data[col].mean()
                if mean != 0:
                    scores.append(min(data[col].std() / abs(mean), 1.0))
                else:
                    scores.append(0.5)
            else:
                probs = data[col].value_counts(normalize=True)
                if len(probs) > 0:
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                    max_entropy = np.log(len(probs) + 1e-10)
                    scores.append(entropy / max_entropy if max_entropy > 0 else 0)
                else:
                    scores.append(0)
        return np.mean(scores) if scores else 0.5
    
    def _privacy_score(self, data: pd.DataFrame) -> float:
        if len(data) == 0:
            return 0
        scores = []
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0
            scores.append(1 - min(unique_ratio, 1.0))
        return np.mean(scores) if scores else 0
    
    def _statistics(self, data: pd.DataFrame) -> dict:
        stats_dict = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'q25': data[col].quantile(0.25),
                    'q50': data[col].quantile(0.50),
                    'q75': data[col].quantile(0.75)
                }
            else:
                stats_dict[col] = {
                    'unique': data[col].nunique(),
                    'most_common': data[col].value_counts().index[0] if len(data[col]) > 0 else None,
                    'most_common_freq': data[col].value_counts().iloc[0] if len(data[col]) > 0 else 0
                }
        return stats_dict
    
    # ==================== ADVANCED METRICS ====================
    
    def calculate_advanced_metrics(self, data: pd.DataFrame) -> dict:
        """Calculate advanced quality metrics"""
        
        metrics = {}
        
        # 1. Data Completeness Score
        total_cells = data.shape[0] * data.shape[1]
        null_cells = data.isnull().sum().sum()
        metrics['completeness'] = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # 2. Column Consistency
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        metrics['type_consistency'] = len([col for col in data.columns 
                                           if data[col].dtype == data[col].dtype]) / len(data.columns) if len(data.columns) > 0 else 0
        
        # 3. Value Range Validity
        if len(numeric_cols) > 0:
            valid_ranges = 0
            for col in numeric_cols:
                mean = data[col].mean()
                std = data[col].std()
                outliers = ((data[col] < mean - 3*std) | (data[col] > mean + 3*std)).sum()
                if outliers / len(data) < 0.05:
                    valid_ranges += 1
            metrics['range_validity'] = valid_ranges / len(numeric_cols)
        else:
            metrics['range_validity'] = 0
        
        # 4. Categorical Diversity
        if len(categorical_cols) > 0:
            diversity_scores = []
            for col in categorical_cols:
                unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0
                diversity_scores.append(unique_ratio)
            metrics['categorical_diversity'] = np.mean(diversity_scores) if diversity_scores else 0
        else:
            metrics['categorical_diversity'] = 0
        
        # 5. Data Freshness (if datetime columns exist)
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            latest_date = data[datetime_cols[0]].max()
            if pd.notna(latest_date):
                days_old = (datetime.now() - latest_date).days
                metrics['data_freshness'] = max(0, 1 - (days_old / 365))
            else:
                metrics['data_freshness'] = 0
        else:
            metrics['data_freshness'] = 0
        
        # 6. Overall Quality Score (weighted)
        weights = {
            'completeness': 0.25,
            'type_consistency': 0.15,
            'range_validity': 0.20,
            'categorical_diversity': 0.20,
            'data_freshness': 0.20
        }
        
        overall = 0
        for metric, weight in weights.items():
            if metric in metrics:
                overall += metrics[metric] * weight
        metrics['overall_quality'] = overall
        
        return metrics


# ==================== DIFFERENTIAL PRIVACY ====================

class DifferentialPrivacy:
    """Apply differential privacy to synthetic data"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(self, data: pd.DataFrame, sensitivity: float = 1.0) -> pd.DataFrame:
        """Add Laplace noise for differential privacy"""
        df = data.copy()
        scale = sensitivity / self.epsilon
        
        for col in df.select_dtypes(include=[np.number]).columns:
            noise = np.random.laplace(0, scale, len(df))
            df[col] = df[col] + noise
        
        return df
    
    def add_gaussian_noise(self, data: pd.DataFrame, sensitivity: float = 1.0) -> pd.DataFrame:
        """Add Gaussian noise for differential privacy"""
        df = data.copy()
        scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        for col in df.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, scale, len(df))
            df[col] = df[col] + noise
        
        return df
    
    def k_anonymity(self, data: pd.DataFrame, quasi_identifiers: list, k: int = 5) -> pd.DataFrame:
        """Apply k-anonymity to data"""
        df = data.copy()
        
        for col in quasi_identifiers:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.cut(df[col], bins=k, labels=[f'Group_{i}' for i in range(k)])
                else:
                    value_counts = df[col].value_counts()
                    rare_values = value_counts[value_counts < k].index
                    df.loc[df[col].isin(rare_values), col] = 'Other'
        
        return df


# ==================== SMART RECOMMENDATIONS ====================

def get_smart_recommendations(data: pd.DataFrame, report: dict) -> list:
    """Generate smart recommendations based on data quality"""
    
    recommendations = []
    
    # Check completeness
    if report.get('completeness', 1) < 0.9:
        null_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
        recommendations.append({
            'severity': 'high',
            'message': f'Missing values detected in {len(null_cols)} columns.',
            'action': 'Use median for numeric, mode for categorical imputation',
            'columns': null_cols[:5]
        })
    
    # Check uniqueness
    if report.get('uniqueness', 0) < 0.1:
        recommendations.append({
            'severity': 'medium',
            'message': 'Low uniqueness detected. Data may have many duplicates.',
            'action': 'Remove duplicate rows or add more varied data'
        })
    
    # Check for potential PII
    pii_keywords = ['email', 'phone', 'address', 'ssn', 'credit', 'card', 'id']
    pii_cols = [col for col in data.columns if any(kw in col.lower() for kw in pii_keywords)]
    if pii_cols:
        recommendations.append({
            'severity': 'critical',
            'message': f'Potential PII detected in: {", ".join(pii_cols[:3])}',
            'action': 'Apply anonymization or differential privacy',
            'columns': pii_cols
        })
    
    # Check for numeric outliers
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:
        mean = data[col].mean()
        std = data[col].std()
        outliers = ((data[col] < mean - 3*std) | (data[col] > mean + 3*std)).sum()
        if outliers / len(data) > 0.05:
            recommendations.append({
                'severity': 'medium',
                'message': f'Outliers detected in column: {col} ({outliers} records)',
                'action': 'Consider winsorization or removal of extreme values'
            })
    
    return recommendations


# ==================== EXPORT WITH METADATA ====================

def export_with_metadata(df: pd.DataFrame, filename: str, format: str = "csv") -> bytes:
    """Export data with metadata"""
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'record_count': len(df),
        'column_count': len(df.columns),
        'columns': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add data file
        if format == "csv":
            data_bytes = df.to_csv(index=False).encode()
            zip_file.writestr('data.csv', data_bytes)
        elif format == "json":
            data_bytes = df.to_json(orient='records', indent=2).encode()
            zip_file.writestr('data.json', data_bytes)
        elif format == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            zip_file.writestr('data.xlsx', output.getvalue())
        
        # Add metadata
        zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
    
    return zip_buffer.getvalue()


# ==================== PLACEHOLDER FUNCTIONS ====================
# These will be implemented when integrating with app.py

def generate_personal_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_sales_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_employee_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_timeseries_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_logs_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_system_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_iot_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_healthcare_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_financial_data(n):
    """Placeholder - will be imported from generators"""
    pass

def generate_toll_data(n):
    """Placeholder - will be imported from generators"""
    pass


# ==================== CACHING ====================

import streamlit as st

@st.cache_data(ttl=3600)
def cached_generate(data_type: str, num_records: int) -> pd.DataFrame:
    """Cache generated data for faster loading"""
    
    # Import generators dynamically to avoid circular imports
    from generators.template import TemplateGenerator
    
    generator = TemplateGenerator()
    return generator.generate(data_type, num_records)