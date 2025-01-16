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


