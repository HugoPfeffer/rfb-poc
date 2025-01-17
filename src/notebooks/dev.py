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
# Load metadata
metadata_path = Path(__file__).parent.parent / "config" / "ctgan_metadata" / "dividas_e_onus.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# %%
# Create SingleTableMetadata
table_metadata = SingleTableMetadata()
for column, sdtype in metadata['sdtypes'].items():
    table_metadata.add_column(column_name=column, sdtype=sdtype)

# %%
# Initialize CTGAN with metadata
synthesizer = CTGANSynthesizer(
    table_metadata,
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
print("Training CTGAN model...")
synthesizer.fit(df)

# %%
print("Training complete!")
# %%
# Generate 100 synthetic samples
synthetic_data = synthesizer.sample(num_rows=1000)



# %%
# Display first few rows of synthetic data
print("\nFirst few rows of synthetic data:")
synthetic_data
# %%
diagnostic_report = run_diagnostic(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=table_metadata)

# %%
diagnostic_report.get_details(property_name='Data Validity')

# %%
quality_report = evaluate_quality(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=table_metadata)

# %%
quality_report.get_details(property_name='Column Shapes')

# %%
fig = get_column_plot(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=table_metadata,
    column_name='ano_calendario'
)
    
fig.show()
# %%
