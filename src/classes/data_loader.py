import pandas as pd
from pathlib import Path
from typing import Dict

class DataLoader:
    def __init__(self):
        """Initialize DataLoader with project root path."""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self._dataframes: Dict[str, pd.DataFrame] = {}
    
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
    
    def load_single_csv(self, filename: str, encoding: str = 'utf-8', delimiter: str = None) -> pd.DataFrame:
        """Load a single CSV file by filename with automatic delimiter detection."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path)
            
        try:
            # First attempt with detected delimiter
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
                    
                    return pd.DataFrame([line.split(delimiter) for line in lines[1:]], 
                                      columns=lines[0].split(delimiter))
