# %%
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from classes.data_loader import DataLoader

# %%
loader = DataLoader()

# %%
df = loader.load_single_csv("Bens e Direitos.csv")

# %%
df.head()

################################################
# load_selected_csvs()
# %%
selected_df_list = loader.load_selected_csvs(["Bens e Direitos.csv", "dividas-e-onus.csv"])

# note that it load_selected_csvs returns a dictionary with the dataframes as keys

################################################
# load_all_csvs()
all_df_dict = loader.load_all_csvs()

# note that it load_all_csvs returns a dictionary with the dataframes as keys
# %%
all_df_dict.keys()
# %%
bens_e_direitos_df = all_df_dict["Bens e Direitos.csv"]