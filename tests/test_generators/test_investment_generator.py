import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime
import sys
from typing import Dict, Any, Optional

from src.generators.investment_generator import InvestmentDataGenerator
from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger

class TestInvestmentGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = InvestmentDataGenerator()
        self.test_settings = self.generator.test_settings
        self.test_size = self.test_settings['default_test_size']
        
    def test_initialization(self):
        """Test if the generator initializes correctly."""
        self.assertIsNotNone(self.generator.employment_generator)
        self.assertIsInstance(self.generator.fraud_probabilities, dict)
        self.assertIsInstance(self.generator.investment_rates, dict)
        self.assertIsInstance(self.generator.asset_classes, dict)
        
    def test_generate_dataset(self):
        """Test if the generator creates a valid dataset with all required columns."""
        df = self.generator.generate(self.test_size)
        
        # Check required columns from settings
        required_columns = set(self.test_settings['validation']['required_columns']['investment'])
        
        self.assertEqual(len(df), self.test_size)
        self.assertTrue(all(col in df.columns for col in required_columns))
        
    def test_fraud_scenarios(self):
        """Test if fraud scenarios are generated with reasonable proportions."""
        # Use a very large sample size for stable proportions
        sample_size = 10000  # Larger sample for more stable statistics
        df = self.generator.generate(sample_size)
        
        # Check fraud type distributions
        fraud_counts = df['fraud_type'].value_counts(normalize=True)
        
        # Get expected probability of non-fraud
        expected_non_fraud = 1 - self.generator.config['fraud_scenarios']['probability']
        
        # Use a wider tolerance for random generation
        tolerance = 0.02  # 2% tolerance for random variation
        
        # Verify fraud probabilities are within reasonable bounds
        self.assertTrue(
            abs(fraud_counts['none'] - expected_non_fraud) < tolerance,
            f"Non-fraud proportion {fraud_counts['none']:.3f} differs from expected {expected_non_fraud:.3f} by more than {tolerance}"
        )
        
        # Verify fraud indicators are consistent
        self.assertTrue(all(
            df[df['fraud_type'] != 'none']['is_fraudulent']
        ))
        
    def test_asset_allocation(self):
        """Test if asset allocations are valid."""
        df = self.generator.generate(self.test_size)
        
        # Check allocation sums to 1
        allocation_cols = ['stocks', 'bonds', 'cash', 'real_estate']
        allocation_sums = df[allocation_cols].sum(axis=1)
        
        self.assertTrue(all(np.isclose(allocation_sums, 1.0, atol=self.test_settings['asset_allocation']['atol'])))
        
        # Check allocations are within bounds from settings
        allocation_ranges = {
            'stocks': self.test_settings['asset_allocation']['stocks_range'],
            'bonds': self.test_settings['asset_allocation']['bonds_range'],
            'cash': self.test_settings['asset_allocation']['cash_range'],
            'real_estate': self.test_settings['asset_allocation']['real_estate_range']
        }
        
        for asset, (min_alloc, max_alloc) in allocation_ranges.items():
            self.assertTrue(all(
                (df[asset] >= min_alloc) & (df[asset] <= max_alloc)
            ), f"Asset {asset} allocations outside allowed range [{min_alloc}, {max_alloc}]")
            
    def test_lifestyle_indicators(self):
        """Test if lifestyle indicators are generated correctly."""
        df = self.generator.generate(self.test_size)
        
        # Check lifestyle ratio calculation
        calculated_ratio = (df['luxury_spending'] + df['travel_spending']) / df['reported_salary']
        calculated_ratio = round(calculated_ratio, 4)
        calculated_ratio.name = 'lifestyle_ratio'  # Set the series name to match
        
        pd.testing.assert_series_equal(
            calculated_ratio,
            df['lifestyle_ratio']
        )
        
        # Verify suspicious lifestyle flag
        expected_suspicious = (
            (df['lifestyle_ratio'] > self.test_settings['lifestyle_thresholds']['ratio_threshold']) |
            (df['luxury_spending'] > df['reported_salary'] * self.test_settings['lifestyle_thresholds']['luxury_spending_ratio'])
        )
        expected_suspicious.name = 'suspicious_lifestyle'  # Set the series name to match
        
        pd.testing.assert_series_equal(
            df['suspicious_lifestyle'],
            expected_suspicious
        )

if __name__ == '__main__':
    unittest.main() 