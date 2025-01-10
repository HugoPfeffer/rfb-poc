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

from src.controllers.dataset_generator import DatasetGenerator

class TestDatasetGenerator(DatasetGenerator):
    """Concrete implementation of DatasetGenerator for testing."""
    
    def generate(self, size: int) -> pd.DataFrame:
        """Generate a simple test dataset."""
        self.data = pd.DataFrame({
            'id': range(size),
            'value': np.random.rand(size),
            'category': np.random.choice(['A', 'B', 'C'], size=size)
        })
        return self.data
    
    def validate(self) -> bool:
        """Simple validation for testing."""
        if self.data is None:
            return False
        return all(col in self.data.columns for col in ['id', 'value', 'category'])
    
    def add_fraud_scenarios(self) -> None:
        """Add simple fraud scenarios for testing."""
        if self.data is not None:
            self.data['is_fraudulent'] = np.random.choice(
                [True, False], 
                size=len(self.data), 
                p=[0.1, 0.9]
            )

class TestDatasetGeneratorClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = TestDatasetGenerator()
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
        self.assertIsNone(self.generator.data)
        self.assertEqual(self.generator._datasets, {})
        self.assertIsInstance(self.generator._metadata, dict)
        
    def test_generate_and_validate(self):
        """Test data generation and validation."""
        df = self.generator.generate(self.test_size)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.test_size)
        self.assertTrue(self.generator.validate())
        
    def test_save_formats(self):
        """Test saving data in different formats with metadata."""
        self.generator.generate(self.test_size)
        
        # Test CSV format
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.generator.save(csv_path, format='csv', include_metadata=True)
        self.assertTrue(os.path.exists(csv_path))
        metadata_path = Path(csv_path).with_suffix('.metadata.json')
        self.assertTrue(os.path.exists(metadata_path))
        
        # Test Parquet format
        parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.generator.save(parquet_path, format='parquet')
        self.assertTrue(os.path.exists(parquet_path))
        
        # Test JSON format
        json_path = os.path.join(self.temp_dir, 'test.json')
        self.generator.save(json_path, format='json')
        self.assertTrue(os.path.exists(json_path))
        
    def test_metadata_content(self):
        """Test metadata generation and content."""
        self.generator.generate(self.test_size)
        metadata = self.generator.get_metadata()
        
        self.assertIsInstance(metadata['created_at'], datetime)
        self.assertEqual(metadata['records'], self.test_size)
        self.assertEqual(set(metadata['columns']), {'id', 'value', 'category'})
        self.assertTrue(metadata['validation_status'])
        
    def test_save_to_dataset(self):
        """Test saving and retrieving datasets in memory."""
        df = self.generator.generate(self.test_size)
        
        # Save dataset
        saved_df = self.generator.save_to_dataset('test_dataset')
        self.assertTrue(pd.DataFrame.equals(df, saved_df))
        
        # Retrieve dataset
        retrieved_df = self.generator.get_dataset('test_dataset')
        self.assertTrue(pd.DataFrame.equals(df, retrieved_df))
        
        # Test non-existent dataset
        with self.assertRaises(KeyError):
            self.generator.get_dataset('non_existent')
            
    def test_config_handling(self):
        """Test configuration loading and retrieval."""
        # Create test config
        config_path = os.path.join(self.temp_dir, 'test_config.json')
        test_config = {'test_key': 'test_value'}
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
            
        # Test config loading
        self.generator.load_config(config_path)
        self.assertEqual(
            self.generator.get_config('test_key'),
            'test_value'
        )
        
        # Test default value
        self.assertEqual(
            self.generator.get_config('non_existent', 'default'),
            'default'
        )
        
    def test_describe_dataset(self):
        """Test dataset description functionality."""
        self.generator.generate(self.test_size)
        
        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        self.generator.describe_dataset()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        # Check for key statistical information instead of specific formatting
        self.assertIn('count', output)
        self.assertIn('mean', output)
        self.assertIn('std', output)
        self.assertIn('min', output)
        self.assertIn('max', output)
        
    def test_sample_data(self):
        """Test data sampling functionality."""
        self.generator.generate(self.test_size)
        
        # Test default sample size
        sample = self.generator.sample_data()
        self.assertEqual(len(sample), 5)
        
        # Test custom sample size
        sample = self.generator.sample_data(n=10)
        self.assertEqual(len(sample), 10)
        
        # Test sample size larger than dataset
        sample = self.generator.sample_data(n=self.test_size + 10)
        self.assertEqual(len(sample), self.test_size)
        
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test saving without data
        with self.assertRaises(ValueError):
            self.generator.save('test.csv')
            
        # Test invalid format
        self.generator.generate(self.test_size)
        with self.assertRaises(ValueError):
            self.generator.save('test.invalid', format='invalid')
            
        # Test sampling without data
        generator = TestDatasetGenerator()
        with self.assertRaises(ValueError):
            generator.sample_data()

if __name__ == '__main__':
    unittest.main() 