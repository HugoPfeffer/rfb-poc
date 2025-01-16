import json
import os
from pathlib import Path
import pandas as pd
import csv

class DataLoader:
    def __init__(self, config_path: str = "src/config/dev/settings.json"):
        """Initialize DataLoader with path to config file.
        
        Args:
            config_path (str): Path to the settings.json file
        """
        # Get the project root directory (2 levels up from this file)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_path = self.project_root / config_path
        self.settings = self._load_settings()
        
    def _load_settings(self) -> dict:
        """Load settings from JSON file.
        
        Returns:
            dict: Settings dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Settings file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in settings file: {self.config_path}")
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """Detect the delimiter used in the CSV file.
        
        Args:
            file_path (Path): Path to the CSV file
            
        Returns:
            str: Detected delimiter
        """
        # Common delimiters to check
        delimiters = [',', ';', '|', '\t']
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first line to analyze
                first_line = f.readline()
                
                # Try each delimiter and count occurrences
                counts = {delimiter: first_line.count(delimiter) for delimiter in delimiters}
                
                # Get delimiter with maximum occurrences
                max_delimiter = max(counts.items(), key=lambda x: x[1])
                
                # If we found a delimiter with more than 0 occurrences, return it
                if max_delimiter[1] > 0:
                    return max_delimiter[0]
                
                return ','  # Default to comma if no other delimiter found
                
        except Exception as e:
            print(f"Warning: Error detecting delimiter: {e}")
            return ','
    
    def load_data(self, encoding: str = 'utf-8', delimiter: str = None) -> pd.DataFrame:
        """Load CSV file into DataFrame using path from settings.
        
        Args:
            encoding (str): File encoding to use (default: utf-8)
            delimiter (str): Optional delimiter to use. If None, will attempt to detect
            
        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        csv_path = self.project_root / self.settings['data']['csv_path']
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        # Detect delimiter if not provided
        if delimiter is None:
            delimiter = self._detect_delimiter(csv_path)
            
        try:
            # First attempt: direct read with detected delimiter
            return pd.read_csv(csv_path, encoding=encoding, delimiter=delimiter)
        except Exception as e:
            print(f"Warning: Initial read failed: {e}")
            try:
                # Second attempt: try with different encoding
                return pd.read_csv(csv_path, encoding='latin1', delimiter=delimiter)
            except Exception as e:
                print(f"Warning: Second attempt failed: {e}")
                # Final attempt: use csv module to handle problematic files
                with open(csv_path, 'r', encoding=encoding) as f:
                    # Read a few lines to determine header and data
                    lines = [line.strip() for line in f.readlines()]
                    if not lines:
                        raise ValueError("Empty file")
                    
                    # Create DataFrame from the parsed data
                    df = pd.DataFrame([line.split(delimiter) for line in lines[1:]], 
                                    columns=lines[0].split(delimiter))
                    return df

    @property
    def data_path(self) -> str:
        """Get the configured data path.
        
        Returns:
            str: Path to CSV file
        """
        return str(self.project_root / self.settings['data']['csv_path']) 