from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from .dataset_generator import DatasetGenerator
from .employment_generator import EmploymentGenerator

class InvestmentGenerator(DatasetGenerator):
    """Generator for synthetic investment data with fraud scenarios."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the investment data generator."""
        super().__init__(config)
        
        self.employment_generator = EmploymentGenerator(config)
        
        # Fraud scenario probabilities
        self.fraud_probabilities = {
            'underreported_income': 0.15,    # 15% chance of underreporting
            'overstated_deductions': 0.12,   # 12% chance of overstating
            'lifestyle_mismatch': 0.10       # 10% chance of lifestyle mismatch
        }
        
        # Investment parameters
        self.investment_rates = {
            'entry': (0.05, 0.15),
            'mid': (0.10, 0.25),
            'senior': (0.15, 0.35)
        }
        
        self.asset_classes = {
            'stocks': (0.4, 0.8),
            'bonds': (0.1, 0.4),
            'cash': (0.05, 0.2),
            'real_estate': (0, 0.2)
        }
        
        # Lifestyle indicators
        self.lifestyle_indicators = {
            'luxury_purchases': {
                'entry': (0, 0.1),      # 0-10% of salary
                'mid': (0.05, 0.15),    # 5-15% of salary
                'senior': (0.1, 0.25)   # 10-25% of salary
            },
            'travel_expenses': {
                'entry': (0.02, 0.08),  # 2-8% of salary
                'mid': (0.05, 0.12),    # 5-12% of salary
                'senior': (0.08, 0.20)  # 8-20% of salary
            }
        }

    def generate(self, size: int) -> pd.DataFrame:
        """Generate synthetic investment dataset with fraud scenarios."""
        # Generate base employment data
        employment_data = self.employment_generator.generate(size)
        self.data = employment_data.copy()
        
        # Add investment data
        self.add_portfolio_data()
        self.add_returns_data()
        
        # Add fraud scenarios and lifestyle indicators
        self.add_fraud_scenarios()
        self.add_lifestyle_indicators()
        
        return self.data
    
    def add_fraud_scenarios(self) -> None:
        """Add fraud scenarios to the dataset."""
        size = len(self.data)
        
        # Initialize fraud indicator columns
        self.data['is_fraudulent'] = False
        self.data['fraud_type'] = 'none'
        self.data['reported_income'] = self.data['salary'].astype('float64')
        self.data['reported_deductions'] = pd.Series(np.zeros(size), dtype='float64')
        self.data['true_deductions'] = pd.Series(np.zeros(size), dtype='float64')
        
        # Generate fraud scenarios
        for idx in range(size):
            if np.random.random() < self.fraud_probabilities['underreported_income']:
                # Underreported income scenario
                true_income = float(self.data.loc[idx, 'salary'])
                underreport_factor = np.random.uniform(0.6, 0.85)
                self.data.loc[idx, 'reported_income'] = round(true_income * underreport_factor, 2)
                self.data.loc[idx, 'is_fraudulent'] = True
                self.data.loc[idx, 'fraud_type'] = 'underreported_income'
                
            elif np.random.random() < self.fraud_probabilities['overstated_deductions']:
                # Overstated deductions scenario
                salary = float(self.data.loc[idx, 'salary'])
                true_deductions = round(salary * np.random.uniform(0.1, 0.2), 2)
                inflated_factor = np.random.uniform(1.3, 1.8)
                self.data.loc[idx, 'true_deductions'] = true_deductions
                self.data.loc[idx, 'reported_deductions'] = round(true_deductions * inflated_factor, 2)
                self.data.loc[idx, 'is_fraudulent'] = True
                self.data.loc[idx, 'fraud_type'] = 'overstated_deductions'
                
    def add_lifestyle_indicators(self) -> None:
        """Add lifestyle indicators to help identify potential fraud."""
        size = len(self.data)
        luxury_spending = np.zeros(size)
        travel_spending = np.zeros(size)
        
        for idx, row in self.data.iterrows():
            exp_level = row['experience_level']
            salary = float(row['salary'])
            
            # Generate luxury purchases
            min_lux, max_lux = self.lifestyle_indicators['luxury_purchases'][exp_level]
            luxury_spending[idx] = round(salary * np.random.uniform(min_lux, max_lux), 2)
            
            # Generate travel expenses
            min_travel, max_travel = self.lifestyle_indicators['travel_expenses'][exp_level]
            travel_spending[idx] = round(salary * np.random.uniform(min_travel, max_travel), 2)
            
            # Add lifestyle mismatch scenarios
            if np.random.random() < self.fraud_probabilities['lifestyle_mismatch']:
                # Inflate lifestyle spending for some fraud cases
                inflation_factor = np.random.uniform(1.5, 2.5)
                if not self.data.loc[idx, 'is_fraudulent']:
                    luxury_spending[idx] *= inflation_factor
                    travel_spending[idx] *= inflation_factor
                    self.data.loc[idx, 'is_fraudulent'] = True
                    self.data.loc[idx, 'fraud_type'] = 'lifestyle_mismatch'
        
        # Add spending columns
        self.data['luxury_spending'] = pd.Series(luxury_spending, dtype='float64')
        self.data['travel_spending'] = pd.Series(travel_spending, dtype='float64')
        
        # Add lifestyle ratio indicators
        lifestyle_ratio = (self.data['luxury_spending'] + self.data['travel_spending']) / self.data['reported_income']
        self.data['lifestyle_ratio'] = round(lifestyle_ratio, 4)
        
        # Add suspicious activity flags
        self.data['suspicious_lifestyle'] = (
            (self.data['lifestyle_ratio'] > 0.4) |  # Spending >40% of reported income
            (self.data['luxury_spending'] > self.data['reported_income'] * 0.25)  # Luxury >25% of income
        )

    def validate(self) -> bool:
        """Validate the generated investment dataset."""
        if self.data is None:
            return False
            
        required_columns = {
            'industry', 'experience_level', 'salary', 'annual_investment',
            'stocks', 'bonds', 'cash', 'real_estate', 
            'expected_annual_return', 'expected_value_1yr',
            'reported_income', 'reported_deductions', 'true_deductions',
            'luxury_spending', 'travel_spending', 'lifestyle_ratio',
            'is_fraudulent', 'fraud_type', 'suspicious_lifestyle'
        }
        
        # Check required columns
        if not all(col in self.data.columns for col in required_columns):
            return False
            
        # Validate allocations sum to 1
        allocation_cols = ['stocks', 'bonds', 'cash', 'real_estate']
        allocations_sum = self.data[allocation_cols].sum(axis=1)
        if not np.allclose(allocations_sum, 1.0, atol=1e-4):
            return False
            
        # Validate fraud scenarios
        if not all(self.data['reported_income'] <= self.data['salary']):
            return False
            
        if not all(self.data['reported_deductions'] >= self.data['true_deductions']):
            return False
            
        return True

    def add_portfolio_data(self) -> None:
        """Add portfolio information including investment amounts and allocations."""
        # Calculate investment amounts based on salary and experience level
        investment_amounts = []
        
        for _, row in self.data.iterrows():
            exp_level = row['experience_level']
            salary = row['salary']
            
            # Get investment rate range for experience level
            min_rate, max_rate = self.investment_rates[exp_level]
            
            # Calculate random investment rate within range
            investment_rate = np.random.uniform(min_rate, max_rate)
            investment_amount = salary * investment_rate
            investment_amounts.append(round(investment_amount, 2))
        
        self.data['annual_investment'] = investment_amounts
        
        # Generate asset allocations
        allocations = []
        for _ in range(len(self.data)):
            allocation = self._generate_allocation()
            allocations.append(allocation)
            
        # Add allocation columns
        allocation_df = pd.DataFrame(allocations)
        self.data = pd.concat([self.data, allocation_df], axis=1)
    
    def _generate_allocation(self) -> Dict[str, float]:
        """Generate random asset allocation that sums to 100%.
        
        Returns:
            Dict[str, float]: Asset allocation percentages
            
        Raises:
            ValueError: If unable to generate valid allocation after max attempts
        """
        MAX_ATTEMPTS = 10
        for attempt in range(MAX_ATTEMPTS):
            try:
                allocation = {}
                remaining = 1.0
                assets = list(self.asset_classes.items())
                
                # Handle all but the last asset
                for asset, (min_alloc, max_alloc) in assets[:-1]:
                    # Calculate maximum possible allocation considering remaining assets' minimums
                    remaining_min_sum = sum(min_a for _, (min_a, _) in assets[assets.index((asset, (min_alloc, max_alloc)))+1:])
                    max_possible = min(max_alloc, remaining - remaining_min_sum)
                    
                    if max_possible < min_alloc:
                        raise ValueError("Cannot satisfy minimum allocation")
                    
                    # Generate allocation within bounds
                    alloc = np.random.uniform(min_alloc, max_possible)
                    allocation[asset] = round(alloc, 4)
                    remaining -= alloc
                
                # Handle last asset
                last_asset, (min_alloc, max_alloc) = assets[-1]
                if not (min_alloc <= remaining <= max_alloc):
                    raise ValueError("Invalid remaining allocation for last asset")
                
                allocation[last_asset] = round(remaining, 4)
                
                # Verify total allocation and bounds
                total = sum(allocation.values())
                if not np.isclose(total, 1.0, atol=1e-4):
                    raise ValueError(f"Total allocation {total} not 100%")
                
                # Verify all allocations are within bounds
                for asset, (min_alloc, max_alloc) in self.asset_classes.items():
                    if not (min_alloc <= allocation[asset] <= max_alloc):
                        raise ValueError(f"Allocation for {asset} outside bounds")
                
                return allocation
                
            except ValueError as e:
                if attempt == MAX_ATTEMPTS - 1:
                    raise ValueError(f"Failed to generate valid asset allocation: {str(e)}")
                continue
    
    def add_returns_data(self) -> None:
        """Add investment returns based on asset allocations."""
        # Define average annual returns and volatility for each asset class
        asset_returns = {
            'stocks': (0.10, 0.15),    # 10% return, 15% volatility
            'bonds': (0.05, 0.05),     # 5% return, 5% volatility
            'cash': (0.02, 0.01),      # 2% return, 1% volatility
            'real_estate': (0.07, 0.10) # 7% return, 10% volatility
        }
        
        # Calculate portfolio returns
        portfolio_returns = []
        
        for _, row in self.data.iterrows():
            total_return = 0
            for asset, (avg_return, volatility) in asset_returns.items():
                allocation = row[asset]
                # Generate random return with normal distribution
                asset_return = np.random.normal(avg_return, volatility)
                total_return += allocation * asset_return
            
            portfolio_returns.append(round(total_return, 4))
        
        self.data['expected_annual_return'] = portfolio_returns
        self.data['expected_value_1yr'] = round(
            self.data['annual_investment'] * (1 + self.data['expected_annual_return']), 
            2
        )
    
    # Keep existing methods: add_portfolio_data(), _generate_allocation(), add_returns_data()
    # ... (previous implementation remains the same) 