# %%
from classes.folder_process import FolderProcess
from load_path import *
import os

# %%
# Initialize
processor = FolderProcess()

# %%
# Create backup and standardize filenames

# %%
mapping = processor.standardize_filenames()
# Example output: "Bens e Direitos.csv" -> "bens_e_direitos.csv"

# %%
# If needed, restore from backup
# processor.restore_from_backup()  # restores most recent
# # or
# processor.restore_from_backup("20240315_143022")

# %%
