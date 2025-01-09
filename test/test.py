import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import os
from datetime import datetime
from src.controllers.employment_generator import EmploymentGenerator
from typing import Any, Dict, Optional

class TestEmploymentGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = EmploymentGenerator()
        self.test_size = 100
        
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
        self.generator.add_industry_data(self.test_size)
        self.generator.add_experience_levels(self.test_size)
        
        # Check if all experience levels are valid
        valid_levels = set(self.generator.experience_levels)
        self.assertTrue(all(level in valid_levels for level in self.generator.data['experience_level']))
        
        # Check approximate distribution (allowing for random variation)
        level_counts = self.generator.data['experience_level'].value_counts(normalize=True)
        self.assertAlmostEqual(level_counts['entry'], 0.4, delta=0.1)
        self.assertAlmostEqual(level_counts['mid'], 0.4, delta=0.1)
        self.assertAlmostEqual(level_counts['senior'], 0.2, delta=0.1)
        
    def test_salary_data(self):
        """Test if salary data is generated within expected ranges."""
        df = self.generator.generate(self.test_size)
        
        for _, row in df.iterrows():
            industry_info = next(ind for ind in self.generator.industry_data 
                               if ind['industry'] == row['industry'])
            base_salary = industry_info['salary_ranges'][row['experience_level']]
            
            # Check if salary is within ±10% of base salary
            self.assertTrue(0.9 * base_salary <= row['salary'] <= 1.1 * base_salary)
            
    def test_validate_method(self):
        """Test the validate method with valid and invalid data."""
        # Test with valid data
        df = self.generator.generate(self.test_size)
        self.assertTrue(self.generator.validate())
        
        # Test with missing data
        self.generator.data = None
        self.assertFalse(self.generator.validate())
        
        # Test with missing columns
        self.generator.data = pd.DataFrame({'industry': ['Test']})
        self.assertFalse(self.generator.validate())
        
    def test_save_method(self):
        """Test if the save method works for different formats."""
        df = self.generator.generate(self.test_size)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV save
            csv_path = os.path.join(tmpdir, 'test.csv')
            self.generator.save(csv_path, format='csv')
            self.assertTrue(os.path.exists(csv_path))
            
            # Test Parquet save
            parquet_path = os.path.join(tmpdir, 'test.parquet')
            self.generator.save(parquet_path, format='parquet')
            self.assertTrue(os.path.exists(parquet_path))
            
            # Test invalid format
            with self.assertRaises(ValueError):
                self.generator.save(csv_path, format='invalid')
                
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test saving without generated data
        generator = EmploymentGenerator()
        with self.assertRaises(ValueError):
            generator.save('test.csv')
            
        # Test invalid config path
        with self.assertRaises(FileNotFoundError):
            generator = EmploymentGenerator({'config_path': 'invalid/path.json'})

    def test_data_sample(self):
        """Display a sample of the generated data for inspection."""
        df = self.generator.generate(self.test_size)
        
        print("\nSample of Generated Employment Data:")
        print("-----------------------------------")
        print("\nFirst 5 records:")
        print(df.head())
        
        print("\nDataset Summary:")
        print("---------------")
        print(f"Total Records: {len(df)}")
        print("\nExperience Level Distribution:")
        print(df['experience_level'].value_counts(normalize=True).round(3))
        
        print("\nSalary Statistics by Experience Level:")
        print(df.groupby('experience_level')['salary'].describe())
        
        print("\nIndustry Distribution:")
        print(df['industry'].value_counts())
        
        # Still need an assertion to make it a valid test
        self.assertTrue(len(df) == self.test_size)

    def test_save_industry_dataset(self):
        """Test if the industry dataset is saved with correct date stamp."""
        # Generate some test data
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
