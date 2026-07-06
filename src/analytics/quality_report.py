"""
Quality Report - Comprehensive Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class QualityReporter:
    """Generate comprehensive quality reports"""
    
    def generate_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate full quality report"""
        
        return {
            'overall_score': self._overall_score(data),
            'completeness': self._completeness(data),
            'uniqueness': self._uniqueness(data),
            'diversity': self._diversity(data),
            'privacy_score': self._privacy_score(data),
            'statistics': self._statistics(data),
            'null_analysis': self._null_analysis(data)
        }
    
    def _overall_score(self, data: pd.DataFrame) -> float:
        scores = [
            self._completeness(data),
            self._uniqueness(data),
            self._diversity(data)
        ]
        return round(sum(scores) / len(scores), 3)
    
    def _completeness(self, data: pd.DataFrame) -> float:
        total = data.shape[0] * data.shape[1]
        nulls = data.isnull().sum().sum()
        return 1 - (nulls / total)
    
    def _uniqueness(self, data: pd.DataFrame) -> float:
        if len(data) == 0:
            return 0
        unique_ratios = []
        for col in data.columns:
            unique_ratios.append(data[col].nunique() / len(data))
        return np.mean(unique_ratios)
    
    def _diversity(self, data: pd.DataFrame) -> float:
        scores = []
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                mean = data[col].mean()
                if mean != 0:
                    cv = data[col].std() / abs(mean)
                    scores.append(min(cv, 1.0))
                else:
                    scores.append(0.5)
            else:
                probs = data[col].value_counts(normalize=True)
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                max_entropy = np.log(len(probs) + 1e-10)
                if max_entropy > 0:
                    scores.append(entropy / max_entropy)
                else:
                    scores.append(0)
        return np.mean(scores) if scores else 0.5
    
    def _privacy_score(self, data: pd.DataFrame) -> float:
        if len(data) == 0:
            return 0
        scores = []
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            scores.append(1 - unique_ratio)
        return np.mean(scores)
    
    def _statistics(self, data: pd.DataFrame) -> Dict:
        stats_dict = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }
            else:
                stats_dict[col] = {
                    'unique': data[col].nunique(),
                    'most_common': data[col].value_counts().index[0] if len(data[col]) > 0 else None
                }
        return stats_dict
    
    def _null_analysis(self, data: pd.DataFrame) -> Dict:
        return {
            'total_nulls': int(data.isnull().sum().sum()),
            'null_by_column': data.isnull().sum().to_dict(),
            'null_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
        }