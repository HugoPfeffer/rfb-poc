# %%
from load_path import *
from classes.data_loader import DataLoader
from classes.process_dataframe import DataFrameProcessor


# %%
# Initialize DataLoader
loader = DataLoader()
# Initialize DataFrameProcessor
processor = DataFrameProcessor()

# %%
# Load all CSVs into dictionary
df_raw = loader.load_single_csv("dividas_e_onus.csv")
df = processor.normalize_columns(df_raw)

# %%
df

# %%
