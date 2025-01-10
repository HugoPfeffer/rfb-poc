import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pytest

from src.generators.components import DataComponent
from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger

class TestDataComponent(DataComponent):
    """Test implementation of DataComponent for testing purposes.
    Note: This is not a test class, but a helper class for testing."""
    
    __test__ = False  # Tell pytest to ignore this class
    
    def generate(self, size: int) -> pd.DataFrame:
        """Generate test data.
        
        Args:
            size: Number of records to generate
            
        Returns:
            pd.DataFrame: Generated test dataset
        """
        return pd.DataFrame({
            'id': range(size),
            'value': self.rng.rand(size)
        })

class TestDatasetGeneratorClass(unittest.TestCase):
    """Test cases for the base DataComponent class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = {
            'random_seed': 42,
            'test_settings': {
                'default_test_size': 100
            }
        }
        self.test_size = self.test_config['test_settings']['default_test_size']
        
    def test_initialization(self):
        """Test if the base component initializes correctly."""
        component = TestDataComponent(self.test_config)
        
        # Check if random seed is set correctly
        self.assertIsNotNone(component.rng)
        self.assertIsInstance(component.rng, np.random.RandomState)
        
        # Check if config is stored correctly
        self.assertEqual(component.config['random_seed'], self.test_config['random_seed'])
        
    def test_generate_method(self):
        """Test if the generate method produces correct output."""
        component = TestDataComponent(self.test_config)
        df = component.generate(self.test_size)
        
        # Check if DataFrame is created with correct size
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.test_size)
        
        # Check if required columns exist
        required_columns = {'id', 'value'}
        self.assertTrue(all(col in df.columns for col in required_columns))
        
    def test_random_state(self):
        """Test if random state produces reproducible results."""
        component1 = TestDataComponent(self.test_config)
        component2 = TestDataComponent(self.test_config)
        
        df1 = component1.generate(self.test_size)
        df2 = component2.generate(self.test_size)
        
        # Check if random seed produces identical results
        pd.testing.assert_frame_equal(df1, df2)
        
    def test_different_seeds(self):
        """Test if different random seeds produce different results."""
        config1 = self.test_config.copy()
        config2 = self.test_config.copy()
        config2['random_seed'] = 43
        
        component1 = TestDataComponent(config1)
        component2 = TestDataComponent(config2)
        
        df1 = component1.generate(self.test_size)
        df2 = component2.generate(self.test_size)
        
        # Check if different seeds produce different results
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(df1, df2)

if __name__ == '__main__':
    unittest.main() 