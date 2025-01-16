import pandas as pd
import unicodedata
import re
from typing import Dict, List, Union

class DataFrameProcessor:
    def __init__(self):
        """Initialize DataFrameProcessor."""
        self.original_columns: Dict[str, str] = {}  # original_name -> normalized_name
        
    def _normalize_column_name(self, column: str) -> str:
        """Normalize a single column name.
        
        Args:
            column (str): Original column name
            
        Returns:
            str: Normalized column name
        """
        # Convert to lowercase
        name = str(column).lower().strip()
        
        # Normalize special characters (é -> e, ç -> c, etc)
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-z0-9]+', '_', name)
        
        # Remove leading/trailing underscores and collapse multiple underscores
        name = re.sub(r'_+', '_', name).strip('_')
        
        return name
    
    def normalize_columns(self, df: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """Normalize all column names in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            inplace (bool): If True, modify the DataFrame in place
            
        Returns:
            pd.DataFrame or None: If inplace=False, returns a new DataFrame with normalized columns.
                                If inplace=True, returns None and modifies the input DataFrame.
        """
        # Store original column mapping
        self.original_columns = {col: self._normalize_column_name(col) for col in df.columns}
        
        # Create new column names
        new_columns = [self.original_columns[col] for col in df.columns]
        
        if inplace:
            df.columns = new_columns
            return None
        else:
            df_copy = df.copy()
            df_copy.columns = new_columns
            return df_copy
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get the mapping of original to normalized column names.
        
        Returns:
            Dict[str, str]: Mapping of original column names to their normalized versions
        """
        return self.original_columns.copy()
    
    def restore_original_columns(self, df: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """Restore original column names from the stored mapping.
        
        Args:
            df (pd.DataFrame): DataFrame with normalized columns
            inplace (bool): If True, modify the DataFrame in place
            
        Returns:
            pd.DataFrame or None: If inplace=False, returns a new DataFrame with original columns.
                                If inplace=True, returns None and modifies the input DataFrame.
        
        Raises:
            ValueError: If there's no stored column mapping or if columns don't match
        """
        if not self.original_columns:
            raise ValueError("No column mapping found. Did you run normalize_columns first?")
            
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in self.original_columns.items()}
        
        # Check if all current columns exist in the mapping
        missing_cols = set(df.columns) - set(reverse_mapping.keys())
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in original mapping")
            
        # Get original column names in current order
        original_columns = [reverse_mapping[col] for col in df.columns]
        
        if inplace:
            df.columns = original_columns
            return None
        else:
            df_copy = df.copy()
            df_copy.columns = original_columns
            return df_copy 