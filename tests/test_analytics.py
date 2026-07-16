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

    def test_privacy_score_differs_from_id_uniqueness(self):
        data = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
            'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC'],
        })

        reporter = QualityReporter()
        report = reporter.generate_report(data)

        assert report['id_uniqueness'] == 1.0
        assert report['privacy_score'] == pytest.approx(0.2)
        assert report['privacy_score'] != report['id_uniqueness']