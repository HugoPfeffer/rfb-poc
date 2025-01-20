# %%
from load_path import *
from classes.data_loader import DataLoader
from classes.process_dataframe import DataFrameProcessor
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import json
from pathlib import Path
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot


# %%
# Initialize DataLoader and DataFrameProcessor
loader = DataLoader()
processor = DataFrameProcessor()

# %%
# Set NA handling
processor.set_fill_na(True)

# %%
# Load raw data
df_raw = loader.load_single_csv("dividas_e_onus.csv")

# %%
# Get normalized column names
normalized_cols = processor.get_columns(df_raw)
print("Normalized columns:", normalized_cols)

# %%
# Process the DataFrame with types
df = processor.normalize_columns(df_raw)

# Create processed_data directory if it doesn't exist
processed_data_path = Path("data/processed_data")
processed_data_path.mkdir(parents=True, exist_ok=True)

# Save processed DataFrame to JSON
output_file = processed_data_path / "dividas_e_onus.json"
df.to_json(output_file, orient='records', force_ascii=False, indent=4)
print(f"DataFrame saved to {output_file}")

# %%
