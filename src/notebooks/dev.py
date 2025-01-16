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
all_df_dict = loader.load_all_csvs()

# %%
all_df_dict.keys()

# %%
bens_e_direitos_df = all_df_dict["dividas_e_onus.csv"]
bens_e_direitos_df = processor.normalize_columns(bens_e_direitos_df)


# %%
bens_e_direitos_df.head()