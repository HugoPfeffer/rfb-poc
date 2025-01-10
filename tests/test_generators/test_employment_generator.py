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
import scipy.stats

from src.generators.employment_generator import EmploymentDataGenerator
from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger

class TestEmploymentGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = EmploymentDataGenerator()
        self.test_settings = self.generator.config['test_settings']
        self.test_size = self.test_settings['default_test_size']
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
        self.assertIsNotNone(self.generator.config)
        self.assertIsInstance(self.generator.rng, np.random.RandomState)
        
    def test_generate_dataset(self):
        """Test if the generator creates a valid dataset of the correct size."""
        df = self.generator.generate(self.test_size)
        
        # Check if DataFrame is created with correct size
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.test_size)
        
        # Check if all required columns exist using settings
        required_columns = set(self.test_settings['validation']['required_columns']['employment'])
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
        test_size = self.test_settings['large_test_size']
        self.generator.add_industry_data(test_size)
        self.generator.add_experience_levels(test_size)
        
        # Check if all experience levels are valid
        valid_levels = set(self.generator.experience_levels)
        self.assertTrue(all(level in valid_levels for level in self.generator.data['experience_level']))
        
        # Get expected probabilities from settings
        expected_probs = {
            level: self.generator.config['experience_levels'][level]['probability']
            for level in self.generator.experience_levels
        }
        
        # Check approximate distribution (allowing for random variation)
        level_counts = self.generator.data['experience_level'].value_counts(normalize=True)
        
        for level, expected_prob in expected_probs.items():
            self.assertAlmostEqual(level_counts[level], expected_prob, delta=self.test_settings['delta_tolerance'])
        
    def test_salary_data(self):
        """Test if salary data is generated with appropriate distribution including outliers."""
        test_size = self.test_settings['large_test_size']
        df = self.generator.generate(test_size)
        
        # Get distribution settings
        dist_settings = self.generator.config['salary_distribution']
        normal_range = dist_settings['normal_range']
        low_outlier = dist_settings['low_outlier']
        high_outlier = dist_settings['high_outlier']
        
        normal_range_count = 0
        outlier_count = 0
        invalid_salaries = 0
        
        print("\nSalary Distribution Analysis:")
        print("----------------------------")
        
        for idx, row in df.iterrows():
            industry_info = next(ind for ind in self.generator.industry_data 
                               if ind['industry'] == row['industry'])
            salary_range = industry_info['salary_ranges'][row['experience_level']]
            min_salary, max_salary = map(float, salary_range.split('-'))
            base_salary = (min_salary + max_salary) / 2
            
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
                    invalid_salaries += 1
                    print(f"\nInvalid salary found: {invalid_salaries}")
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
        print(f"Invalid salaries found: {invalid_salaries}")
        print(f"Normal ratio: {normal_range_count/test_size:.2%}")
        print(f"Outlier ratio: {outlier_count/test_size:.2%}")
        
        # Perform chi-square test for distribution goodness of fit
        observed = np.array([normal_range_count, outlier_count])
        expected_probs = np.array([
            normal_range['probability'],
            low_outlier['probability'] + high_outlier['probability']
        ])
        expected = test_size * expected_probs
        
        chi2, p_value = scipy.stats.chisquare(observed, expected)
        print(f"\nChi-square test results:")
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"p-value: {p_value:.4f}")
        
        # Test should pass if p-value is above significance level (0.05)
        self.assertGreater(
            p_value, 
            0.05, 
            f"Salary distribution failed chi-square test (p={p_value:.4f})"
        )
        
        return invalid_salaries
        
    def test_save_industry_dataset(self):
        """Test if the industry dataset is saved with correct date stamp."""
        df = self.generator.generate(self.test_size)
        
        # Create test output directory if it doesn't exist
        test_output_dir = Path(self.test_settings['output_paths']['test_data'])
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
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
                          for col in self.test_settings['validation']['required_columns']['employment']))
        
        # Clean up the test file
        os.remove(output_path)

if __name__ == '__main__':
    unittest.main() 