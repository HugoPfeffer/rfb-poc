from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import pandas as pd
from functools import wraps

from src.utils.logging_config import app_logger
from src.config.config_manager import config_manager, ConfigurationError

@dataclass
class ValidationError(Exception):
    """Custom exception for validation errors."""
    message: str
    details: Optional[Dict[str, Any]] = None

class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the data.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if validation passes, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        pass

class DataFrameValidationStrategy(ValidationStrategy):
    """Validation strategy for pandas DataFrames."""
    
    def __init__(self, required_columns: List[str], column_types: Optional[Dict[str, Type]] = None):
        self.required_columns = required_columns
        self.column_types = column_types or {}

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate DataFrame structure and content."""
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame")
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}",
                {'missing_columns': list(missing_columns)}
            )
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col in data.columns:
                if not pd.api.types.is_dtype_equal(data[col].dtype, expected_type):
                    raise ValidationError(
                        f"Invalid type for column {col}. Expected {expected_type}, got {data[col].dtype}",
                        {'column': col, 'expected_type': str(expected_type), 'actual_type': str(data[col].dtype)}
                    )
        
        return True

class ConfigValidationStrategy(ValidationStrategy):
    """Validation strategy for configuration dictionaries."""
    
    def __init__(self, required_keys: List[str], key_types: Optional[Dict[str, Type]] = None):
        self.required_keys = required_keys
        self.key_types = key_types or {}

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate configuration structure and types."""
        if not isinstance(data, dict):
            raise ValidationError("Data must be a dictionary")
        
        # Check required keys
        missing_keys = set(self.required_keys) - set(data.keys())
        if missing_keys:
            raise ValidationError(
                f"Missing required configuration keys: {missing_keys}",
                {'missing_keys': list(missing_keys)}
            )
        
        # Check value types
        for key, expected_type in self.key_types.items():
            if key in data:
                if not isinstance(data[key], expected_type):
                    raise ValidationError(
                        f"Invalid type for key {key}. Expected {expected_type}, got {type(data[key])}",
                        {'key': key, 'expected_type': str(expected_type), 'actual_type': str(type(data[key]))}
                    )
        
        return True

def validate_input(validation_strategy: ValidationStrategy):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Assume first non-self argument is the data to validate
                data = args[1] if len(args) > 1 else kwargs.get('data')
                if data is None:
                    raise ValidationError("No data provided for validation")
                
                validation_strategy.validate(data)
                return func(*args, **kwargs)
            except ValidationError as e:
                app_logger.error(f"Validation error in {func.__name__}: {e.message}", extra={'details': e.details})
                raise
        return wrapper
    return decorator

def validate_config(config_keys: List[str], key_types: Optional[Dict[str, Type]] = None):
    """Decorator for configuration validation."""
    strategy = ConfigValidationStrategy(config_keys, key_types)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                strategy.validate(config_manager.config)
                return func(*args, **kwargs)
            except ValidationError as e:
                app_logger.error(f"Configuration validation error in {func.__name__}: {e.message}", 
                               extra={'details': e.details})
                raise
        return wrapper
    return decorator 