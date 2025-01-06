import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define the number of samples to generate
NUM_SAMPLES = 1000
FRAUD_PERCENTAGE = 0.05  # 5% of data will be fraudulent
OUTLIER_PERCENTAGE = 0.02  # 2% of data will be outliers

# Define possible categorical values
FILING_STATUSES = ['Single', 'Married Filing Jointly', 'Married Filing Separately', 'Head of Household']
STATES = [
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
    'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
    'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
]

# Add a dictionary for state names (optional, for reference)
STATE_NAMES = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}

def inject_outliers(data, percentage=OUTLIER_PERCENTAGE):
    """Inject outliers into the dataset."""
    num_outliers = int(len(data) * percentage)
    outlier_indices = np.random.choice(len(data), num_outliers, replace=False)
    
    outlier_data = data.copy()
    
    for idx in outlier_indices:
        # Randomly choose which feature to modify
        feature = np.random.choice(['income', 'deductions', 'tax_paid', 'refund_claimed'])
        
        if feature == 'income':
            # Extremely high income
            outlier_data.loc[idx, 'income'] *= np.random.uniform(10, 20)
        elif feature == 'deductions':
            # Unusually high deductions compared to income
            outlier_data.loc[idx, 'deductions'] = outlier_data.loc[idx, 'income'] * np.random.uniform(0.8, 0.95)
        elif feature == 'tax_paid':
            # Suspiciously low tax paid
            outlier_data.loc[idx, 'tax_paid'] *= np.random.uniform(0.1, 0.2)
        elif feature == 'refund_claimed':
            # Extremely high refund claims
            outlier_data.loc[idx, 'refund_claimed'] = outlier_data.loc[idx, 'tax_paid'] * np.random.uniform(1.5, 2.0)
    
    return outlier_data

def generate_fraudulent_patterns(data, percentage=FRAUD_PERCENTAGE):
    """Generate fraudulent patterns in the dataset."""
    num_fraud = int(len(data) * percentage)
    fraud_indices = np.random.choice(len(data), num_fraud, replace=False)
    
    fraud_data = data.copy()
    fraud_data['is_fraudulent'] = 0  # Initialize fraud indicator column
    
    for idx in fraud_indices:
        fraud_type = np.random.choice([
            'income_underreporting',
            'deduction_inflation',
            'refund_manipulation',
            'compliance_manipulation'
        ])
        
        if fraud_type == 'income_underreporting':
            # Underreport income but maintain high deductions
            fraud_data.loc[idx, 'income'] *= np.random.uniform(0.4, 0.6)
            fraud_data.loc[idx, 'compliance_score'] *= np.random.uniform(0.5, 0.7)
        
        elif fraud_type == 'deduction_inflation':
            # Inflate deductions relative to income
            fraud_data.loc[idx, 'deductions'] = fraud_data.loc[idx, 'income'] * np.random.uniform(0.7, 0.9)
            fraud_data.loc[idx, 'compliance_score'] *= np.random.uniform(0.6, 0.8)
        
        elif fraud_type == 'refund_manipulation':
            # Claim excessive refunds
            fraud_data.loc[idx, 'refund_claimed'] = fraud_data.loc[idx, 'tax_paid'] * np.random.uniform(1.2, 1.8)
            fraud_data.loc[idx, 'compliance_score'] *= np.random.uniform(0.4, 0.6)
        
        elif fraud_type == 'compliance_manipulation':
            # Multiple suspicious patterns
            fraud_data.loc[idx, 'income'] *= np.random.uniform(0.6, 0.8)
            fraud_data.loc[idx, 'deductions'] = fraud_data.loc[idx, 'income'] * np.random.uniform(0.6, 0.8)
            fraud_data.loc[idx, 'compliance_score'] *= np.random.uniform(0.3, 0.5)
        
        fraud_data.loc[idx, 'is_fraudulent'] = 1
    
    return fraud_data

def generate_base_data():
    """Generate initial synthetic tax data."""
    np.random.seed(42)
    
    data = {
        'income': np.random.lognormal(11, 0.7, NUM_SAMPLES),
        'filing_status': np.random.choice(FILING_STATUSES, NUM_SAMPLES),
        'state': np.random.choice(STATES, NUM_SAMPLES),
        'compliance_score': np.random.beta(8, 2, NUM_SAMPLES) * 100,
    }
    
    # Calculate dependent variables with more realistic patterns
    data['deductions'] = np.where(
        data['income'] > np.median(data['income']),
        data['income'] * np.random.beta(2, 5, NUM_SAMPLES),  # Higher income brackets
        data['income'] * np.random.beta(1.5, 6, NUM_SAMPLES)  # Lower income brackets
    )
    
    data['tax_paid'] = np.where(
        data['income'] > np.median(data['income']),
        data['income'] * np.random.beta(3, 7, NUM_SAMPLES),  # Higher income brackets
        data['income'] * np.random.beta(2, 8, NUM_SAMPLES)  # Lower income brackets
    )
    
    data['refund_claimed'] = np.where(
        np.random.random(NUM_SAMPLES) < 0.7,
        data['tax_paid'] * np.random.beta(2, 5, NUM_SAMPLES),
        0
    )
    
    return pd.DataFrame(data)

def train_synthesizer(data):
    """Train the SDV synthesizer on the data."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    # Update metadata for specific columns
    metadata.update_column(
        column_name='filing_status',
        sdtype='categorical'
    )
    metadata.update_column(
        column_name='state',
        sdtype='categorical'
    )
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    
    return synthesizer

def normalize_and_split_data(synthetic_data, test_size=0.2, random_state=42):
    """
    Normalize the numerical features and split the data into training and test sets.
    """
    # Separate numerical and categorical columns
    numerical_cols = ['income', 'deductions', 'tax_paid', 'refund_claimed', 'compliance_score']
    categorical_cols = ['filing_status', 'state']
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Normalize numerical columns
    normalized_data = synthetic_data.copy()
    normalized_data[numerical_cols] = scaler.fit_transform(synthetic_data[numerical_cols])
    
    # Split the data
    train_data, test_data = train_test_split(
        normalized_data,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save the scaler for future use
    joblib.dump(scaler, 'tax_data_scaler.joblib')
    
    return train_data, test_data, scaler

def generate_synthetic_data(num_samples=1000):
    """Generate and save synthetic tax data."""
    # Generate initial data
    base_data = generate_base_data()
    
    # Train synthesizer
    synthesizer = train_synthesizer(base_data)
    
    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_samples)
    
    # Post-process the data
    synthetic_data['income'] = synthetic_data['income'].round(2)
    synthetic_data['deductions'] = synthetic_data['deductions'].round(2)
    synthetic_data['tax_paid'] = synthetic_data['tax_paid'].round(2)
    synthetic_data['refund_claimed'] = synthetic_data['refund_claimed'].round(2)
    synthetic_data['compliance_score'] = synthetic_data['compliance_score'].round(2)
    
    # Inject fraudulent patterns
    synthetic_data = generate_fraudulent_patterns(synthetic_data)
    
    # Inject outliers
    synthetic_data = inject_outliers(synthetic_data)
    
    # Save raw data to CSV
    synthetic_data.to_csv('synthetic_tax_data.csv', index=False)
    
    # Normalize and split the data
    train_data, test_data, scaler = normalize_and_split_data(synthetic_data)
    
    # Save train and test sets
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    
    return synthetic_data, train_data, test_data, scaler

if __name__ == "__main__":
    synthetic_data, train_data, test_data, scaler = generate_synthetic_data()
    print("Synthetic data generated and saved to 'synthetic_tax_data.csv'")
    print(f"Training data saved to 'train_data.csv' (shape: {train_data.shape})")
    print(f"Test data saved to 'test_data.csv' (shape: {test_data.shape})")
    
    # Print fraud statistics
    fraud_count = synthetic_data['is_fraudulent'].sum()
    total_count = len(synthetic_data)
    print(f"\nFraud Statistics:")
    print(f"Total records: {total_count}")
    print(f"Fraudulent records: {fraud_count} ({(fraud_count/total_count)*100:.2f}%)")
    
    print("\nSample of data with fraud indicators:")
    print(synthetic_data[synthetic_data['is_fraudulent'] == 1].head())
