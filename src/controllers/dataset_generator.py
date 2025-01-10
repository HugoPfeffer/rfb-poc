from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetGenerator(ABC):
    """Base class for all dataset generators.
    
    This class provides the core functionality and interface that all dataset
    generators must implement. It handles common operations like saving datasets,
    validation, configuration management, and data preprocessing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset generator.
        
        Args:
            config: Configuration dictionary for the generator. If None, loads from default settings.json
        """
        self.config = config or {}
        self.settings = self.config.copy()  # Create a copy to avoid modifying the original
        
        if not self.settings:
            settings_path = Path(__file__).parent.parent.parent / 'data' / 'configs' / 'settings.json'
            try:
                with open(settings_path, 'r') as f:
                    self.settings = json.load(f)
                    self.config = self.settings.copy()  # Keep original config in sync
            except FileNotFoundError:
                logger.warning(f"Settings file not found at {settings_path}, using default settings")
                self._init_default_settings()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in settings file at {settings_path}, using default settings")
                self._init_default_settings()
        
        self.data: Optional[pd.DataFrame] = None
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Any] = {
            'created_at': None,
            'records': 0,
            'columns': [],
            'data_types': {},
            'validation_status': False
        }
        
        # Initialize random state if seed is provided
        self._init_random_state()
        
    def _init_default_settings(self) -> None:
        """Initialize default settings if settings file is not available."""
        self.settings = {
            'random_seed': 42,
            'test_settings': {
                'default_test_size': 100,
                'large_test_size': 1000,
                'delta_tolerance': 0.05,
                'sample_size': {
                    'default': 5,
                    'max': 10
                },
                'temp_file_formats': ['csv', 'parquet', 'json'],
                'validation': {
                    'required_columns': {
                        'base': ['id', 'value', 'category'],
                        'employment': ['industry', 'experience_level', 'salary'],
                        'investment': [
                            'industry', 'experience_level', 'salary', 'annual_investment',
                            'stocks', 'bonds', 'cash', 'real_estate', 
                            'expected_annual_return', 'expected_value_1yr',
                            'reported_income', 'reported_deductions', 'true_deductions',
                            'luxury_spending', 'travel_spending', 'lifestyle_ratio',
                            'is_fraudulent', 'fraud_type', 'suspicious_lifestyle'
                        ]
                    }
                },
                'output_paths': {
                    'test_data': 'data/test',
                    'temp': 'data/temp'
                }
            }
        }
        self.config = self.settings.copy()
        
    def _init_random_state(self) -> None:
        """Initialize random state from settings."""
        random_seed = self.settings.get('random_seed', 42)  # Use main random_seed
        np.random.seed(random_seed)
        logger.debug(f"Initialized random state with seed: {random_seed}")
    
    @abstractmethod
    def generate(self, size: int) -> pd.DataFrame:
        """Generate synthetic dataset.
        
        Args:
            size: Number of records to generate
            
        Returns:
            pd.DataFrame: Generated synthetic dataset
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the generated dataset.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        pass
    
    @abstractmethod
    def add_fraud_scenarios(self) -> None:
        """Add fraud scenarios to the dataset.
        
        This method should be implemented by subclasses to add
        domain-specific fraud patterns.
        """
        pass
    
    def save(self, path: str, format: str = 'csv', include_metadata: bool = True) -> None:
        """Save the generated dataset to disk.
        
        Args:
            path: Path where to save the dataset
            format: Format to save the data in (csv, parquet, etc.)
            include_metadata: Whether to save metadata alongside the dataset
            
        Raises:
            ValueError: If data hasn't been generated or format is unsupported
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data has been generated yet")
            
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata before saving
        self._update_metadata()
        
        # Save data in specified format
        if format.lower() == 'csv':
            self.data.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            self.data.to_parquet(output_path, index=False)
        elif format.lower() == 'json':
            self.data.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Save metadata if requested
        if include_metadata:
            metadata_path = output_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self._metadata, f, indent=2, default=str)
    
    def save_to_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Store the generated data in memory and return it as a DataFrame.
        
        Args:
            dataset_name: Name to identify the dataset
            
        Returns:
            pd.DataFrame: The stored dataset
            
        Raises:
            ValueError: If data hasn't been generated
        """
        if self.data is None:
            raise ValueError("No data has been generated yet")
        
        self._datasets[dataset_name] = self.data.copy()
        return self._datasets[dataset_name]
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Retrieve a stored dataset by name.
        
        Args:
            dataset_name: Name of the dataset to retrieve
            
        Returns:
            pd.DataFrame: The stored dataset
            
        Raises:
            KeyError: If dataset_name doesn't exist
        """
        if dataset_name not in self._datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        return self._datasets[dataset_name]
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config.update(json.load(f))
            
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        return self.config.get(key, default)
    
    def _update_metadata(self) -> None:
        """Update metadata about the generated dataset."""
        if self.data is not None:
            self._metadata.update({
                'created_at': datetime.now(),
                'records': len(self.data),
                'columns': list(self.data.columns),
                'data_types': self.data.dtypes.astype(str).to_dict(),
                'validation_status': self.validate(),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'null_counts': self.data.isnull().sum().to_dict()
            })
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the generated dataset.
        
        Returns:
            Dict containing dataset metadata
        """
        self._update_metadata()
        return self._metadata
    
    def describe_dataset(self) -> None:
        """Print a comprehensive description of the dataset."""
        if self.data is None:
            logger.warning("No data has been generated yet")
            return
            
        logger.info("\nDataset Description:")
        logger.info("-" * 20)
        
        # Basic information
        logger.info(f"\nRecords: {len(self.data)}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
        # Data types
        logger.info("\nData Types:")
        for col, dtype in self.data.dtypes.items():
            logger.info(f"{col}: {dtype}")
        
        # Summary statistics
        logger.info("\nSummary Statistics:")
        print(self.data.describe())
        
        # Null values
        logger.info("\nNull Values:")
        null_counts = self.data.isnull().sum()
        if null_counts.any():
            print(null_counts[null_counts > 0])
        else:
            logger.info("No null values found")
    
    def sample_data(self, n: int = 5) -> pd.DataFrame:
        """Get a random sample of the dataset.
        
        Args:
            n: Number of records to sample
            
        Returns:
            DataFrame containing sampled records
        """
        if self.data is None:
            raise ValueError("No data has been generated yet")
        return self.data.sample(n=min(n, len(self.data)))
