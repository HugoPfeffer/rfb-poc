# %%
import numpy as np
from scipy.stats.mstats import winsorize
import scipy.stats as stats
import pandas as pd


class InvestmentsGenerator:
    def __init__(self, mean_return=0.24, std_dev=0.10, sample_size=1000):
        """
        Initialize the InvestmentsGenerator with default parameters.
        
        Args:
            mean_return (float): Expected mean annual return (default: 0.24 or 24%)
            std_dev (float): Standard deviation of returns (default: 0.10 or 10%)
            sample_size (int): Number of samples to generate (default: 1000)
        """
        self.mean_return = mean_return
        self.std_dev = std_dev
        self.sample_size = sample_size
        # Add sectors data
        self.sectors = [
            {"setor_id": 1, "setor_nome": "Tecnologia da Informação"},
            {"setor_id": 2, "setor_nome": "Saúde"},
            {"setor_id": 3, "setor_nome": "Finanças"},
            # Add more sectors as needed
        ]
        
    def generate_basic_returns(self):
        """Generate basic synthetic returns using normal distribution."""
        return np.random.normal(loc=self.mean_return, 
                              scale=self.std_dev, 
                              size=self.sample_size)
    
    def clean_data(self, data, limits=[0.01, 0.01]):
        """
        Winsorize data to remove extreme outliers.
        
        Args:
            data (np.array): Input data to clean
            limits (list): Lower and upper percentile limits for winsorization
        """
        return winsorize(data, limits=limits)
    
    def generate_with_crisis(self, crisis_prob=0.1):
        """
        Generate returns with varying volatility to simulate crisis periods.
        Returns a DataFrame with sector information and returns.
        """
        returns = np.random.normal(
            loc=self.mean_return, 
            scale=np.random.choice(
                [self.std_dev, self.std_dev * 3],
                size=self.sample_size,
                p=[1 - crisis_prob, crisis_prob]
            )
        )
        
        # Create DataFrame with sectors and returns
        df = pd.DataFrame(self.sectors)
        df['returns'] = returns[:len(self.sectors)]  # Trim returns to match sectors length
        return df






