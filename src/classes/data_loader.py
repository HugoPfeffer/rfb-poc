import pandas as pd
from pathlib import Path
from typing import Dict, List
from .dataset_validation import DatasetValidation

class DataLoader:
    def __init__(self):
        """Initialize DataLoader with project root path."""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self.validator = DatasetValidation()
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """Detect the delimiter used in the CSV file."""
        delimiters = [',', ';', '|', '\t']
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                counts = {delimiter: first_line.count(delimiter) for delimiter in delimiters}
                max_delimiter = max(counts.items(), key=lambda x: x[1])
                
                return max_delimiter[0] if max_delimiter[1] > 0 else ','
                
        except Exception as e:
            print(f"Warning: Error detecting delimiter: {e}")
            return ','
    
    def _load_csv_file(self, file_path: Path, encoding: str = 'utf-8-sig', delimiter: str = None) -> pd.DataFrame:
        """Internal method to load a CSV file with error handling."""
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path)
            
        try:
            # First attempt with detected delimiter and utf-8-sig encoding
            return pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        except Exception as e:
            print(f"Warning: Initial read failed: {e}")
            try:
                # Second attempt with different encoding
                return pd.read_csv(file_path, encoding='latin1', delimiter=delimiter)
            except Exception as e:
                print(f"Warning: Second attempt failed: {e}")
                # Final attempt using manual parsing
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f.readlines()]
                    if not lines:
                        raise ValueError("Empty file")
                    
                    # Remove BOM from header if present
                    headers = lines[0].split(delimiter)
                    headers = [h.strip('\ufeff') for h in headers]
                    
                    return pd.DataFrame([line.split(delimiter) for line in lines[1:]], 
                                      columns=headers)
    
    def load_single_csv(self, filename: str, encoding: str = 'utf-8', delimiter: str = None, validate: bool = True) -> pd.DataFrame:
        """Load a single CSV file by filename with automatic delimiter detection and validation.
        
        Args:
            filename (str): Name of the CSV file to load
            encoding (str): File encoding to use
            delimiter (str, optional): Specific delimiter to use. If None, will detect automatically
            validate (bool): Whether to validate and print validation results
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        df = self._load_csv_file(file_path, encoding, delimiter)
        
        # Store in cache
        self._dataframes[filename] = df
        
        # Run validation if requested
        if validate:
            self.validator.validate_dataset(df, filename)
            self.validator.print_validation_results(filename)
        
        return df
    
    def load_selected_csvs(self, filenames: List[str], encoding: str = 'utf-8', 
                          delimiter: str = None, validate: bool = True) -> Dict[str, pd.DataFrame]:
        """Load multiple selected CSV files with validation."""
        results = {}
        for filename in filenames:
            try:
                df = self.load_single_csv(filename, encoding, delimiter, validate)
                results[filename] = df
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
        return results
    
    def load_all_csvs(self, encoding: str = 'utf-8', delimiter: str = None, validate: bool = True) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the data directory with validation."""
        csv_files = [f.name for f in self.data_dir.glob("*.csv")]
        if not csv_files:
            print(f"Warning: No CSV files found in {self.data_dir}")
            return {}
            
        return self.load_selected_csvs(csv_files, encoding, delimiter, validate)
    
    def get_cached_files(self) -> List[str]:
        """Get list of currently cached files.
        
        Returns:
            List[str]: List of filenames that are currently cached
        """
        return list(self._dataframes.keys())
