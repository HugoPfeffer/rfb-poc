import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import os
from datetime import datetime
import sys
from typing import Dict, Any, Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.controllers.employment_generator import EmploymentGenerator

class TestEmploymentGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set fixed random seed for reproducible tests
        np.random.seed(42)
        self.generator = EmploymentGenerator()
        self.test_size = 100
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary files
        for file in Path(self.temp_dir).glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)
        
    def test_initialization(self):
        """Test if the generator initializes correctly."""
        self.assertIsNotNone(self.generator.industry_data)
        self.assertEqual(self.generator.experience_levels, ['entry', 'mid', 'senior'])
        
    def test_generate_dataset(self):
        """Test if the generator creates a valid dataset of the correct size."""
        df = self.generator.generate(self.test_size)
        
        # Check if DataFrame is created with correct size
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.test_size)
        
        # Check if all required columns exist
        required_columns = {'industry', 'experience_level', 'salary'}
        self.assertTrue(all(col in df.columns for col in required_columns))
        
    def test_industry_data(self):
        """Test if industry data is generated correctly."""
        self.generator.add_industry_data(self.test_size)
        
        # Check if industries are from the configuration
        valid_industries = {ind['industry'] for ind in self.generator.industry_data}
        self.assertTrue(all(ind in valid_industries for ind in self.generator.data['industry']))
        
    def test_experience_levels(self):
        """Test if experience levels are generated with correct distribution."""
        # Use a larger sample size for more stable distribution
        test_size = 1000
        self.generator.add_industry_data(test_size)
        self.generator.add_experience_levels(test_size)
        
        # Check if all experience levels are valid
        valid_levels = set(self.generator.experience_levels)
        self.assertTrue(all(level in valid_levels for level in self.generator.data['experience_level']))
        
        # Check approximate distribution (allowing for random variation)
        level_counts = self.generator.data['experience_level'].value_counts(normalize=True)
        
        # With fixed seed and larger sample size, we can use tighter bounds
        self.assertAlmostEqual(level_counts['entry'], 0.4, delta=0.05)
        self.assertAlmostEqual(level_counts['mid'], 0.4, delta=0.05)
        self.assertAlmostEqual(level_counts['senior'], 0.2, delta=0.05)
        
    def test_salary_data(self):
        """Test if salary data is generated within expected ranges."""
        df = self.generator.generate(self.test_size)
        
        for _, row in df.iterrows():
            industry_info = next(ind for ind in self.generator.industry_data 
                               if ind['industry'] == row['industry'])
            base_salary = industry_info['salary_ranges'][row['experience_level']]
            
            # Check if salary is within ±10% of base salary
            self.assertTrue(0.9 * base_salary <= row['salary'] <= 1.1 * base_salary)
            
    def test_save_industry_dataset(self):
        """Test if the industry dataset is saved with correct date stamp."""
        df = self.generator.generate(self.test_size)
        
        # Save the dataset
        output_path = self.generator.save_industry_dataset()
        
        # Verify the file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Verify the filename format
        filename = os.path.basename(output_path)
        current_date = datetime.now().strftime('%Y%m%d')
        expected_filename = f'industry_dataset_{current_date}.csv'
        self.assertEqual(filename, expected_filename)
        
        # Verify the file content
        saved_df = pd.read_csv(output_path)
        self.assertEqual(len(saved_df), self.test_size)
        self.assertTrue(all(col in saved_df.columns 
                          for col in ['industry', 'experience_level', 'salary']))
        
        # Clean up the test file
        os.remove(output_path)

if __name__ == '__main__':
    unittest.main() 