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

    def test_numeric_defaults_to_integers(self):
        sample = pd.DataFrame({
            'quantity': [1, 2, 1, 2, 3],
            'score': [10.5, 11.2, 9.8, 10.1, 11.0],
        })
        gen = CustomGenerator(sample)
        df = gen.generate(20)
        assert all(float(v).is_integer() for v in df['quantity'])
        assert all(float(v).is_integer() for v in df['score'])

    def test_price_columns_use_two_decimal_places_max(self):
        sample = pd.DataFrame({
            'unit_price_usd': [12.99, 267.9046, 148.2547, 47.6817, 75.3189],
            'quantity': [1, 2, 1, 1, 2],
        })
        gen = CustomGenerator(sample)
        df = gen.generate(20)

        for value in df['unit_price_usd']:
            rounded = round(float(value), 2)
            assert abs(float(value) - rounded) < 1e-9

        assert all(float(v).is_integer() for v in df['quantity'])