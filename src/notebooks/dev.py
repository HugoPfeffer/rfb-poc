# %%
from load_path import *
from classes.data_loader import DataLoader
from classes.process_dataframe import DataFrameProcessor
from sdv.single_table import CTGANSynthesizer
import json
from pathlib import Path


# %%
# Initialize DataLoader and DataFrameProcessor
loader = DataLoader()
processor = DataFrameProcessor()

# %%
# Load metadata
metadata_path = Path(__file__).parent.parent / "config" / "ctgan_metadata" / "dividas_e_onus.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# %%
# Initialize CTGAN with metadata
synthesizer = CTGANSynthesizer(
    metadata.get('sdtypes'),
    enforce_min_max_values=True,
    enforce_rounding=True,
    epochs=1000
)

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
# Get original column names
original_cols = processor.get_columns(df_raw, normalized=False)
print("Original columns:", original_cols)

# %%
# Set up column data types
column_dtypes = {
    'ano_calendario': 'float64',
    'emprestimos_contraidos_no_exterior': 'float64',
    'estabelecimento_bancario_comercial': 'float64',
    'outras_dividas_e_onus_reais': 'float64',
    'outras_pessoas_juridicas': 'float64',
    'pessoas_fisicas': 'float64',
    'soc_de_credito_financiamento_e_investimento': 'float64',
    'outros': 'float64',
    'invalido': 'float64'
}
processor.set_column_dtypes(column_dtypes)

# %%
# Process the DataFrame with types
df = processor.normalize_columns(df_raw)

# %%
# Check data types of all columns
print("\nDataframe dtypes:")
df.info()

# %%
df
# %%
