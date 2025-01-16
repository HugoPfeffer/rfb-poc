# %%
from load_path import *
from classes.data_loader import DataLoader

# %%
loader = DataLoader()

# %%
df = loader.load_single_csv("Bens e Direitos.csv")

# %%
df.head()

# %%
selected_df_list = loader.load_selected_csvs(["Bens e Direitos.csv", "dividas-e-onus.csv"])

# %%
all_df_dict = loader.load_all_csvs()

# %%
all_df_dict.keys()

# %%
bens_e_direitos_df = all_df_dict["Bens e Direitos.csv"]

# %%
bens_e_direitos_df.head()
