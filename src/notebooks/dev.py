# %%
from load_path import *
from classes.data_loader import DataLoader
from classes.process_dataframe import DataFrameProcessor


# %%
# Initialize DataLoader and DataFrameProcessor
loader = DataLoader()
processor = DataFrameProcessor()

# %%
# Load raw data
df_raw = loader.load_single_csv("dividas_e_onus.csv")

# %%
# Process the DataFrame
df = processor.normalize_columns(df_raw)

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
# Check data types of all columns
print("\nDataframe dtypes:")
df.info()
