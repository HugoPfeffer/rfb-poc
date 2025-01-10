import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime
import sys
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))
from src.controllers.investment_generator import InvestmentGenerator

class TestInvestmentGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)  # For reproducibility
        self.generator = InvestmentGenerator()
        self.test_size = 100
        
    def test_initialization(self):
        """Test if the generator initializes correctly."""
        self.assertIsNotNone(self.generator.employment_generator)
        self.assertIsInstance(self.generator.fraud_probabilities, dict)
        self.assertIsInstance(self.generator.investment_rates, dict)
        self.assertIsInstance(self.generator.asset_classes, dict)
        
    def test_generate_dataset(self):
        """Test if the generator creates a valid dataset with all required columns."""
        df = self.generator.generate(self.test_size)
        
        required_columns = {
            'industry', 'experience_level', 'salary', 'annual_investment',
            'stocks', 'bonds', 'cash', 'real_estate', 
            'expected_annual_return', 'expected_value_1yr',
            'reported_income', 'reported_deductions', 'true_deductions',
            'luxury_spending', 'travel_spending', 'lifestyle_ratio',
            'is_fraudulent', 'fraud_type', 'suspicious_lifestyle'
        }
        
        self.assertEqual(len(df), self.test_size)
        self.assertTrue(all(col in df.columns for col in required_columns))
        
    def test_fraud_scenarios(self):
        """Test if fraud scenarios are generated correctly."""
        df = self.generator.generate(1000)  # Larger sample for stable proportions
        
        # Check fraud type distributions
        fraud_counts = df['fraud_type'].value_counts(normalize=True)
        
        # Verify fraud probabilities are roughly as expected
        self.assertAlmostEqual(
            fraud_counts['underreported_income'],
            self.generator.fraud_probabilities['underreported_income'],
            delta=0.05
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
        
        self.assertTrue(all(np.isclose(allocation_sums, 1.0, atol=1e-4)))
        
        # Check allocations are within bounds
        for asset, (min_alloc, max_alloc) in self.generator.asset_classes.items():
            self.assertTrue(all(
                (df[asset] >= min_alloc) & (df[asset] <= max_alloc)
            ))
            
    def test_lifestyle_indicators(self):
        """Test if lifestyle indicators are generated correctly."""
        df = self.generator.generate(self.test_size)
        
        # Check lifestyle ratio calculation
        calculated_ratio = (df['luxury_spending'] + df['travel_spending']) / df['reported_income']
        calculated_ratio = round(calculated_ratio, 4)
        calculated_ratio.name = 'lifestyle_ratio'  # Set the series name to match
        
        pd.testing.assert_series_equal(
            calculated_ratio,
            df['lifestyle_ratio']
        )
        
        # Verify suspicious lifestyle flag
        expected_suspicious = (
            (df['lifestyle_ratio'] > 0.4) |
            (df['luxury_spending'] > df['reported_income'] * 0.25)
        )
        expected_suspicious.name = 'suspicious_lifestyle'  # Set the series name to match
        
        pd.testing.assert_series_equal(
            df['suspicious_lifestyle'],
            expected_suspicious
        )

if __name__ == '__main__':
    unittest.main() 