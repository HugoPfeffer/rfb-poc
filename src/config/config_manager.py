from pathlib import Path
import json
from typing import Dict, Any, Optional
from functools import lru_cache
import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    message: str

class ConfigManager:
    """Singleton configuration manager for the application."""
    
    _instance = None
    _config: Dict[str, Any] = None
    _config_path: Path = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
            cls._instance._config_path = None
        return cls._instance

    def initialize(self, config_path: Optional[str] = None) -> None:
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file. If None, uses default.
        """
        if config_path:
            self._config_path = Path(config_path)
        else:
            self._config_path = Path(__file__).parent.parent.parent / 'data' / 'configs' / 'settings.json'
        
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if not self._config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self._config_path}")
            
            with open(self._config_path, 'r') as f:
                self._config = json.load(f)
            
            self._validate_config()
            logger.info(f"Configuration loaded successfully from {self._config_path}")
        
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = {
            'test_settings',
            'experience_levels',
            'fraud_scenarios',
            'random_seed'
        }
        
        if not isinstance(self._config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
            
        missing_sections = required_sections - set(self._config.keys())
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary.
        
        Returns:
            Dict[str, Any]: The configuration dictionary
        """
        if self._config is None:
            raise ConfigurationError("Configuration not initialized")
        return dict(self._config)  # Return a copy of the configuration

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        if self._config is None:
            raise ConfigurationError("Configuration not initialized")
        return self._config.get(key, default)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

# Global instance
config_manager = ConfigManager() 