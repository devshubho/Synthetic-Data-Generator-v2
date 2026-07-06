"""
Test Generators
"""

import pytest
import pandas as pd
from src.generators.template import TemplateGenerator
from src.generators.custom import CustomGenerator

class TestTemplateGenerator:
    
    def test_personal_data(self):
        gen = TemplateGenerator()
        df = gen.generate("Personal/Customer Data", 10)
        assert len(df) == 10
        assert len(df.columns) > 5
    
    def test_sales_data(self):
        gen = TemplateGenerator()
        df = gen.generate("Sales Transactions", 10)
        assert len(df) == 10
        assert 'total' in df.columns

class TestCustomGenerator:
    
    def test_custom_generation(self):
        sample = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        gen = CustomGenerator(sample)
        df = gen.generate(10)
        assert len(df) == 10
        assert len(df.columns) == 2