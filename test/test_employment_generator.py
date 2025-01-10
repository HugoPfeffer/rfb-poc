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
        self.assertEqual(set(self.generator.experience_levels), {'entry', 'mid', 'senior'})
        self.assertIsNotNone(self.generator.settings)
        self.assertIsInstance(self.generator.rng, np.random.RandomState)
        
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
        
        # Get expected probabilities from settings
        expected_probs = {
            level: self.generator.settings['experience_levels'][level]['probability']
            for level in self.generator.experience_levels
        }
        
        # Check approximate distribution (allowing for random variation)
        level_counts = self.generator.data['experience_level'].value_counts(normalize=True)
        
        for level, expected_prob in expected_probs.items():
            self.assertAlmostEqual(level_counts[level], expected_prob, delta=0.05)
        
    def test_salary_data(self):
        """Test if salary data is generated with appropriate distribution including outliers."""
        # Use a larger sample size for more stable distribution
        test_size = 1000
        df = self.generator.generate(test_size)
        
        normal_range_count = 0
        outlier_count = 0
        
        # Get range settings from configuration
        dist_settings = self.generator.settings['salary_distribution']
        normal_range = dist_settings['normal_range']
        low_outlier = dist_settings['low_outlier']
        high_outlier = dist_settings['high_outlier']
        
        print("\nSalary Distribution Analysis:")
        print("----------------------------")
        
        for idx, row in df.iterrows():
            industry_info = next(ind for ind in self.generator.industry_data 
                               if ind['industry'] == row['industry'])
            base_salary = float(industry_info['salary_ranges'][row['experience_level']])
            
            # Calculate percentage of base salary
            ratio = float(row['salary']) / base_salary
            
            # First check if it's in the normal range
            if normal_range['min_ratio'] <= ratio <= normal_range['max_ratio']:
                normal_range_count += 1
            else:
                # If not in normal range, verify it's in one of the outlier ranges
                is_valid_outlier = (
                    (low_outlier['min_ratio'] <= ratio <= low_outlier['max_ratio']) or
                    (high_outlier['min_ratio'] <= ratio <= high_outlier['max_ratio'])
                )
                if not is_valid_outlier:
                    print(f"\nInvalid salary found:")
                    print(f"Industry: {row['industry']}")
                    print(f"Experience: {row['experience_level']}")
                    print(f"Base salary: {base_salary}")
                    print(f"Actual salary: {row['salary']}")
                    print(f"Ratio: {ratio:.2%}")
                
                self.assertTrue(
                    is_valid_outlier,
                    f"Salary {row['salary']} ({ratio:.2%} of base) outside expected ranges relative to base {base_salary}"
                )
                outlier_count += 1
        
        # Print distribution summary
        print(f"\nNormal range count: {normal_range_count}")
        print(f"Outlier count: {outlier_count}")
        print(f"Normal ratio: {normal_range_count/test_size:.2%}")
        print(f"Outlier ratio: {outlier_count/test_size:.2%}")
        
        # Check distribution (allowing for some random variation)
        normal_ratio = normal_range_count / test_size
        outlier_ratio = outlier_count / test_size
        
        expected_normal_prob = normal_range['probability']
        expected_outlier_prob = low_outlier['probability'] + high_outlier['probability']
        
        self.assertAlmostEqual(normal_ratio, expected_normal_prob, delta=0.05)
        self.assertAlmostEqual(outlier_ratio, expected_outlier_prob, delta=0.05)
        
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