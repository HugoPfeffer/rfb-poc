# %%
from load_path import *
from classes.data_loader import DataLoader
from classes.process_dataframe import DataFrameProcessor


# %%
# Initialize DataLoader
loader = DataLoader()

# Initialize DataFrameProcessor with column data types
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
processor = DataFrameProcessor(column_dtypes=column_dtypes)

# %%
# Load and process CSV
df_raw = loader.load_single_csv("dividas_e_onus.csv")
df = processor.normalize_columns(df_raw)

# %%
df

# %%
# Check data types of all columns
print("\nDataframe dtypes:")
df.info()

# %%
print(df)

# %%
