from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
import json

class DatasetGenerator(ABC):
    """Base class for all dataset generators.
    
    This class provides the core functionality and interface that all dataset
    generators must implement. It handles common operations like saving datasets,
    validation, and configuration management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset generator.
        
        Args:
            config: Configuration dictionary for the generator
        """
        self.config = config or {}
        self.data: Optional[pd.DataFrame] = None
        self._datasets: Dict[str, pd.DataFrame] = {}
        
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
    def save(self, path: str, format: str = 'csv') -> None:
        """Save the generated dataset to disk.
        
        This method should be implemented by subclasses to handle specific
        saving requirements for each type of dataset.
        
        Args:
            path: Path where to save the dataset
            format: Format to save the data in (csv, parquet, etc.)
            
        Raises:
            ValueError: If data hasn't been generated or format is unsupported
        """
        pass
    
    def save_to_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Store the generated data in memory and return it as a DataFrame.
        
        This method stores the generated data in memory for later access and
        returns it as a DataFrame. The data can be accessed later using the
        get_dataset method.
        
        Args:
            dataset_name: Name to identify the dataset
            
        Returns:
            pd.DataFrame: The stored dataset
            
        Raises:
            ValueError: If data hasn't been generated
        """
        if self.data is None:
            raise ValueError("No data has been generated yet")
        
        # Store the data in memory
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
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file.
        
        The config file should contain a mapping of parameters needed for data generation.
        
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
