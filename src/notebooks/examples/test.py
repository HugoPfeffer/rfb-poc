# %%
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from classes.core.data_loader import DataLoader

# %% Initialize the DataLoader
loader = DataLoader()
print(f"Data will be loaded from: {loader.data_path}")

# %% Load and display the data
try:
    # Try automatic delimiter detection first
    df = loader.load_data()
    print("\nAutomatic delimiter detection succeeded!")
    
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
except Exception as e:
    print(f"\nAutomatic detection failed: {e}")
    print("\nTrying with explicit '|' delimiter...")
    
    try:
        df = loader.load_data(delimiter='|')
        print("\nLoading with '|' delimiter succeeded!")
        
        print("\nDataFrame Info:")
        print(df.info())
        
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"\nError: Failed to load data with both automatic and manual delimiter: {e}")
        print("Please check the file format and try again.")




# %%
