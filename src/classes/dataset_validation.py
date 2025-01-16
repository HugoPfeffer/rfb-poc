import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from pathlib import Path
import unicodedata
import re
import csv

class DatasetValidation:
    def __init__(self):
        """Initialize DatasetValidation."""
        self.validation_results: Dict[str, Dict] = {}
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        
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
    
    def validate_data_types(self, df: pd.DataFrame, filename: str) -> Dict[str, Dict]:
        """Check data types and potential issues in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            filename (str): Name of the file being validated
            
        Returns:
            Dict[str, Dict]: Validation results including type mismatches and null counts
        """
        result = {
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns_with_text': []
        }
        
        # Check monetary columns (those ending with _r or containing R$) for non-numeric values
        monetary_cols = [col for col in df.columns 
                        if col.lower().endswith('_r') or 'r$' in col.lower()]
        
        for col in monetary_cols:
            # Remove currency symbols and spaces, replace comma with dot
            df[col] = df[col].astype(str).str.replace('R$', '').str.strip()
            df[col] = df[col].str.replace('.', '').str.replace(',', '.')
            
            non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
            if non_numeric_mask.any():
                result['numeric_columns_with_text'].append({
                    'column': col,
                    'invalid_rows': df[non_numeric_mask].index.tolist(),
                    'invalid_values': df.loc[non_numeric_mask, col].tolist()
                })
        
        result['is_valid'] = len(result['numeric_columns_with_text']) == 0
        
        if filename in self.validation_results:
            self.validation_results[filename]['data_types'] = result
        else:
            self.validation_results[filename] = {'data_types': result}
        
        return result
    
    def validate_value_ranges(self, df: pd.DataFrame, filename: str) -> Dict[str, Dict]:
        """Check if values are within expected ranges.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            filename (str): Name of the file being validated
            
        Returns:
            Dict[str, Dict]: Validation results including out-of-range values
        """
        result = {
            'out_of_range_values': [],
            'negative_monetary_values': []
        }
        
        # Check monetary columns for negative values
        monetary_cols = [col for col in df.columns 
                        if col.lower().endswith('_r') or 'r$' in col.lower()]
        
        for col in monetary_cols:
            # Clean and convert monetary values
            df[col] = df[col].astype(str).str.replace('R$', '').str.strip()
            df[col] = df[col].str.replace('.', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            negative_mask = (df[col] < 0) & df[col].notna()
            if negative_mask.any():
                result['negative_monetary_values'].append({
                    'column': col,
                    'invalid_rows': df[negative_mask].index.tolist(),
                    'invalid_values': df.loc[negative_mask, col].tolist()
                })
        
        # Check year columns
        year_cols = [col for col in df.columns 
                    if 'ano' in col.lower() or 'year' in col.lower()]
        current_year = pd.Timestamp.now().year
        
        for col in year_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            invalid_years = (df[col] < 1900) | (df[col] > current_year) & df[col].notna()
            if invalid_years.any():
                result['out_of_range_values'].append({
                    'column': col,
                    'invalid_rows': df[invalid_years].index.tolist(),
                    'invalid_values': df.loc[invalid_years, col].tolist()
                })
        
        result['is_valid'] = (len(result['out_of_range_values']) == 0 and 
                            len(result['negative_monetary_values']) == 0)
        
        if filename in self.validation_results:
            self.validation_results[filename]['value_ranges'] = result
        else:
            self.validation_results[filename] = {'value_ranges': result}
        
        return result
    
    def validate_dataset(self, df: pd.DataFrame, filename: str) -> Dict[str, Dict]:
        """Run all validations on the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            filename (str): Name of the file being validated
            
        Returns:
            Dict[str, Dict]: Complete validation results
        """
        results = {
            'loading_integrity': self.validate_loading_integrity(df, filename),
            'data_types': self.validate_data_types(df, filename),
            'value_ranges': self.validate_value_ranges(df, filename)
        }
        
        # Overall validation status
        results['is_valid'] = all(v.get('is_valid', False) for v in results.values())
        
        self.validation_results[filename] = results
        return results
    
    def print_validation_results(self, filename: str) -> None:
        """Print validation results in a user-friendly format.
        
        Args:
            filename (str): Name of the file to print results for
        """
        if filename not in self.validation_results:
            print(f"No validation results found for {filename}")
            return
            
        results = self.validation_results[filename]
        print(f"\n=== Validation Results for {filename} ===")
        
        # Overall status
        is_valid = results.get('is_valid', False)
        print(f"\nOverall Status: {'✓ Valid' if is_valid else '✗ Invalid'}")
        
        # Loading Integrity
        if 'loading_integrity' in results:
            integrity_results = results['loading_integrity']
            print("\nLoading Integrity:")
            
            if not integrity_results['row_count_match']:
                details = integrity_results['details']['row_count']
                print(f"  ✗ Row Count Mismatch:")
                print(f"    - CSV rows: {details['csv_rows']}")
                print(f"    - DataFrame rows: {details['dataframe_rows']}")
                print(f"    - Difference: {details['difference']} rows")
            else:
                print("  ✓ Row count matches")
            
            if integrity_results['data_integrity_issues']:
                print("  ✗ Data Integrity Issues:")
                # Group issues by type
                numeric_issues = [i for i in integrity_results['data_integrity_issues'] if i['type'] == 'numeric_mismatch']
                value_issues = [i for i in integrity_results['data_integrity_issues'] if i['type'] == 'value_mismatch']
                
                if numeric_issues:
                    print(f"    - {len(numeric_issues)} numeric value mismatches")
                    # Show first few examples
                    for issue in numeric_issues[:3]:
                        print(f"      Row {issue['row']}, Column '{issue['column']}':")
                        print(f"      CSV: {issue['csv_value']} → DataFrame: {issue['df_value']}")
                
                if value_issues:
                    print(f"    - {len(value_issues)} text value mismatches")
                    # Show first few examples
                    for issue in value_issues[:3]:
                        print(f"      Row {issue['row']}, Column '{issue['column']}':")
                        print(f"      CSV: {issue['csv_value']} → DataFrame: {issue['df_value']}")
            else:
                print("  ✓ All values match original CSV")
        
        # Data types
        if 'data_types' in results:
            type_results = results['data_types']
            print("\nData Quality:")
            
            # Null values
            null_counts = type_results['null_counts']
            has_nulls = any(count > 0 for count in null_counts.values())
            if has_nulls:
                print("  ! Null Values Found:")
                for col, count in null_counts.items():
                    if count > 0:
                        print(f"    - {col}: {count} nulls")
            else:
                print("  ✓ No null values found")
            
            # Non-numeric values in numeric columns
            if type_results['numeric_columns_with_text']:
                print("  ✗ Invalid Values in Numeric Columns:")
                for issue in type_results['numeric_columns_with_text']:
                    print(f"    - {issue['column']}: {len(issue['invalid_rows'])} invalid values")
            else:
                print("  ✓ All numeric columns contain valid numbers")
        
        # Value ranges
        if 'value_ranges' in results:
            range_results = results['value_ranges']
            print("\nValue Ranges:")
            
            # Negative values
            if range_results['negative_monetary_values']:
                print("  ✗ Negative Values Found in Monetary Columns:")
                for issue in range_results['negative_monetary_values']:
                    print(f"    - {issue['column']}: {len(issue['invalid_rows'])} negative values")
            else:
                print("  ✓ No negative monetary values found")
            
            # Out of range values
            if range_results['out_of_range_values']:
                print("  ✗ Out of Range Values Found:")
                for issue in range_results['out_of_range_values']:
                    print(f"    - {issue['column']}: {len(issue['invalid_rows'])} invalid values")
            else:
                print("  ✓ All values within expected ranges")
        
        print("\n" + "="*50) 
    
    def validate_loading_integrity(self, df: pd.DataFrame, filename: str) -> Dict[str, Dict]:
        """Validate that the DataFrame matches the original CSV data.
        
        This method checks if the data was loaded correctly by comparing:
        1. Row count matches
        2. No data truncation occurred
        3. Special characters were preserved
        4. Numeric values maintained precision
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            filename (str): Name of the CSV file to compare against
            
        Returns:
            Dict[str, Dict]: Validation results comparing CSV to DataFrame
        """
        result = {
            'is_valid': True,
            'row_count_match': True,
            'data_integrity_issues': [],
            'details': {}
        }
        
        file_path = self.data_dir / filename
        
        # First pass: detect delimiter and count rows
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle BOM
            sample = f.readline()
            # Check common delimiters
            delimiters = [',', ';', '|', '\t']
            counts = {d: sample.count(d) for d in delimiters}
            delimiter = max(counts.items(), key=lambda x: x[1])[0]
            
            # Reset file pointer and count rows
            f.seek(0)
            csv_row_count = sum(1 for _ in f) - 1  # subtract header
        
        # Compare row counts
        df_row_count = len(df)
        if csv_row_count != df_row_count:
            result['is_valid'] = False
            result['row_count_match'] = False
            result['details']['row_count'] = {
                'csv_rows': csv_row_count,
                'dataframe_rows': df_row_count,
                'difference': abs(csv_row_count - df_row_count)
            }
        
        # Detailed comparison of values
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle BOM
            csv_reader = csv.reader(f, delimiter=delimiter)
            headers = [h.strip('\ufeff') for h in next(csv_reader)]  # Remove BOM from headers
            
            for row_idx, csv_row in enumerate(csv_reader, start=0):
                if row_idx >= len(df):
                    break
                    
                for col_idx, (csv_value, header) in enumerate(zip(csv_row, headers)):
                    df_value = str(df.iloc[row_idx][header])
                    csv_value = csv_value.strip()
                    
                    # Check for data differences
                    if csv_value != df_value:
                        # For numeric values, check if it's just formatting
                        try:
                            csv_num = float(csv_value.replace(',', '.').replace('R$', '').strip())
                            df_num = float(df_value.replace(',', '.').replace('R$', '').strip())
                            if abs(csv_num - df_num) > 1e-10:  # Allow small floating-point differences
                                result['data_integrity_issues'].append({
                                    'row': row_idx,
                                    'column': header,
                                    'csv_value': csv_value,
                                    'df_value': df_value,
                                    'type': 'numeric_mismatch'
                                })
                                result['is_valid'] = False
                        except ValueError:
                            # For non-numeric values, report exact mismatches
                            if csv_value != df_value:
                                result['data_integrity_issues'].append({
                                    'row': row_idx,
                                    'column': header,
                                    'csv_value': csv_value,
                                    'df_value': df_value,
                                    'type': 'value_mismatch'
                                })
                                result['is_valid'] = False
        
        # Store results
        if filename in self.validation_results:
            self.validation_results[filename]['loading_integrity'] = result
        else:
            self.validation_results[filename] = {'loading_integrity': result}
        
        return result 