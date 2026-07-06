"""
Test Analytics
"""

import pytest
import pandas as pd
from src.analytics.quality_report import QualityReporter

class TestQualityReporter:
    
    def test_report_generation(self):
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        reporter = QualityReporter()
        report = reporter.generate_report(data)
        
        assert 'overall_score' in report
        assert 'completeness' in report
        assert 'uniqueness' in report