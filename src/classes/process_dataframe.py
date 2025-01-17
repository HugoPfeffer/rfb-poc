import pandas as pd
import unicodedata
import re
from typing import Dict, Union, Optional

class DataFrameProcessor:
    def __init__(self, column_dtypes: Optional[Dict[str, str]] = None):
        """Initialize DataFrameProcessor.
        
        Args:
            column_dtypes (Optional[Dict[str, str]]): Dictionary mapping column names to their desired data types.
        """
        self.original_columns: Dict[str, str] = {}  # original_name -> normalized_name
        self.column_dtypes = column_dtypes or {}
    
    def set_column_dtypes(self, column_dtypes: Dict[str, str]) -> None:
        """Update the column data types mapping.
        
        Args:
            column_dtypes (Dict[str, str]): New mapping of column names to their desired data types
        """
        self.column_dtypes = column_dtypes
    
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
    
    def _convert_numeric_value(self, value: str) -> float:
        """Convert string value to numeric, handling Brazilian number format.
        
        Args:
            value (str): Value as string
            
        Returns:
            float: Converted value
        """
        if pd.isna(value):
            return pd.NA
            
        # Convert to string if not already
        value = str(value)
        
        # Remove currency symbol and spaces
        value = value.replace('R$', '').strip()
        
        # Replace comma with dot for decimal separator
        value = value.replace('.', '').replace(',', '.')
        
        try:
            return float(value)
        except ValueError:
            return pd.NA
    
    def _process_column(self, series: pd.Series, dtype: str) -> pd.Series:
        """Process a single column according to its desired data type.
        
        Args:
            series (pd.Series): Column to process
            dtype (str): Desired data type
            
        Returns:
            pd.Series: Processed column
        """
        # For numeric types, convert Brazilian number format first
        if dtype in ['float64', 'float32', 'int64', 'int32']:
            series = series.apply(self._convert_numeric_value)
        
        # Convert to specified type
        return series.astype(dtype)
    
    def normalize_columns(self, df: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """Normalize column names in the DataFrame and convert data types.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            inplace (bool): If True, modify the DataFrame in place
            
        Returns:
            pd.DataFrame or None: If inplace=False, returns a new DataFrame with normalized columns.
                                If inplace=True, returns None and modifies the input DataFrame.
        """
        # Work with a copy if not inplace
        df_result = df if inplace else df.copy()
        
        # Store original column mapping
        self.original_columns = {col: self._normalize_column_name(col) for col in df_result.columns}
        
        # Create new column names
        new_columns = [self.original_columns[col] for col in df_result.columns]
        df_result.columns = new_columns
        
        # Apply specified data types
        if self.column_dtypes:
            for col, dtype in self.column_dtypes.items():
                if col in df_result.columns:
                    df_result[col] = self._process_column(df_result[col], dtype)
        
        if inplace:
            return None
        return df_result
    
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