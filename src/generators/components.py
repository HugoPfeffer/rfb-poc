from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger, log_execution_time
from src.validation.validation_framework import (
    ValidationStrategy,
    DataFrameValidationStrategy,
    validate_input,
    validate_config
)

class DataComponent(ABC):
    """Abstract base class for data generation components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data component.
        
        Args:
            config: Optional configuration dictionary. If None, loads from config manager.
        """
        if config is None:
            # Initialize config manager if not already initialized
            if not hasattr(config_manager, '_config') or config_manager._config is None:
                config_manager.initialize()
            self.config = dict(config_manager._config)  # Get direct dictionary reference
        else:
            self.config = dict(config)  # Make a copy of the provided config
            
        self.rng = np.random.RandomState(self.config.get('random_seed', 42))
    
    @abstractmethod
    def generate(self, size: int) -> pd.DataFrame:
        """Generate component data."""
        pass

class DataPersistence:
    """Component for data persistence operations."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path('data/generated')
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    @log_execution_time(app_logger)
    def save(self, data: pd.DataFrame, name: str, format: str = 'csv') -> Path:
        """Save data to file.
        
        Args:
            data: DataFrame to save
            name: Base name for the file
            format: File format (csv, parquet, json)
            
        Returns:
            Path: Path to saved file
        """
        filename = f"{name}.{format}"
        file_path = self.base_path / filename
        
        if format == 'csv':
            data.to_csv(file_path, index=False)
        elif format == 'parquet':
            data.to_parquet(file_path, index=False)
        elif format == 'json':
            data.to_json(file_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        app_logger.info(f"Saved data to {file_path}")
        return file_path
    
    def load(self, path: Path) -> pd.DataFrame:
        """Load data from file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.json':
            return pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

class DataTransformation:
    """Component for data transformation operations."""
    
    @staticmethod
    def add_noise(data: pd.DataFrame, column: str, scale: float = 0.1) -> pd.DataFrame:
        """Add random noise to numeric column."""
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column {column} must be numeric")
        
        noise = np.random.normal(0, scale, len(data))
        data[column] = data[column] * (1 + noise)
        return data
    
    @staticmethod
    def categorize(data: pd.DataFrame, column: str, bins: List[float], labels: List[str]) -> pd.DataFrame:
        """Categorize numeric column into bins."""
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column {column} must be numeric")
        
        data[f"{column}_category"] = pd.cut(data[column], bins=bins, labels=labels)
        return data

class FraudScenarioGenerator:
    """Component for generating fraud scenarios."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fraud scenario generator.
        
        Args:
            config: Optional configuration dictionary
        """
        if not hasattr(config_manager, '_config') or config_manager._config is None:
            config_manager.initialize()
        self.config = config or config_manager.config
        self.fraud_settings = self.config.get('fraud_scenarios')
        if not self.fraud_settings:
            raise ValueError("Required fraud_scenarios settings not found in config")
        self.rng = np.random.RandomState(self.config.get('random_seed', 42))
    
    def add_fraud_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add fraud indicators to the dataset."""
        size = len(data)
        data['is_fraudulent'] = pd.Series([False] * size, dtype=bool)
        data['fraud_type'] = pd.Series(['none'] * size, dtype=pd.StringDtype())
        
        # Get fraud probability from settings
        fraud_prob = self.fraud_settings.get('probability')
        if fraud_prob is None:
            raise ValueError("Required fraud_scenarios.probability not found in config")
        
        # Generate fraud mask
        fraud_mask = self.rng.random(size) < fraud_prob
        data.loc[fraud_mask, 'is_fraudulent'] = True
        
        # Get fraud type probabilities from settings
        fraud_types = ['salary_misreporting', 'suspicious_lifestyle', 'rapid_transactions']
        fraud_probs = []
        for fraud_type in fraud_types:
            prob = self.fraud_settings.get(fraud_type, {}).get('probability')
            if prob is None:
                raise ValueError(f"Required fraud_scenarios.{fraud_type}.probability not found in config")
            fraud_probs.append(prob)
        
        # Normalize probabilities
        total_prob = sum(fraud_probs)
        norm_probs = [p/total_prob for p in fraud_probs]
        
        # Assign fraud types
        fraud_indices = data[fraud_mask].index
        assigned_types = self.rng.choice(fraud_types, size=len(fraud_indices), p=norm_probs)
        data.loc[fraud_indices, 'fraud_type'] = pd.Series(assigned_types, dtype=pd.StringDtype())
        
        return data 