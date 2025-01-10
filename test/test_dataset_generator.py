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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test dataset generator."""
        super().__init__(config)
        # Ensure we have test settings even if config loading failed
        if 'test_settings' not in self.settings:
            self.settings['test_settings'] = {
                'default_test_size': 100,
                'sample_size': {'default': 5, 'max': 10},
                'temp_file_formats': ['csv', 'parquet', 'json']
            }
    
    def generate(self, size: int) -> pd.DataFrame:
        """Generate a simple test dataset."""
        # Use the random state from parent class
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
        required_columns = self.settings['test_settings']['validation']['required_columns']['base']
        return all(col in self.data.columns for col in required_columns)
    
    def add_fraud_scenarios(self) -> None:
        """Add simple fraud scenarios for testing."""
        if self.data is not None:
            fraud_prob = self.settings.get('fraud_scenarios', {}).get('probability', 0.1)
            self.data['is_fraudulent'] = np.random.choice(
                [True, False], 
                size=len(self.data), 
                p=[fraud_prob, 1 - fraud_prob]
            )

class TestDatasetGeneratorClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = TestDatasetGenerator()
        # Ensure test_settings exists before accessing it
        if not hasattr(self.generator, 'settings') or 'test_settings' not in self.generator.settings:
            raise ValueError("Test settings not properly initialized in generator")
        self.test_settings = self.generator.settings['test_settings']
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
        self.assertIsNone(self.generator.data)
        self.assertEqual(self.generator._datasets, {})
        self.assertIsInstance(self.generator._metadata, dict)
        
    def test_generate_and_validate(self):
        """Test data generation and validation."""
        df = self.generator.generate(self.test_size)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.test_size)
        
        # Check if all required columns exist using settings
        required_columns = set(self.test_settings['validation']['required_columns']['base'])
        self.assertTrue(all(col in df.columns for col in required_columns))
        self.assertTrue(self.generator.validate())
        
    def test_save_formats(self):
        """Test saving data in different formats with metadata."""
        self.generator.generate(self.test_size)
        
        for format in self.test_settings['temp_file_formats']:
            file_path = os.path.join(self.temp_dir, f'test.{format}')
            self.generator.save(file_path, format=format, include_metadata=(format == 'csv'))
            self.assertTrue(os.path.exists(file_path))
            
            if format == 'csv':
                metadata_path = Path(file_path).with_suffix('.metadata.json')
                self.assertTrue(os.path.exists(metadata_path))
                
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
        self.assertEqual(len(sample), self.test_settings['sample_size']['default'])
        
        # Test custom sample size
        sample = self.generator.sample_data(n=self.test_settings['sample_size']['max'])
        self.assertEqual(len(sample), self.test_settings['sample_size']['max'])
        
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