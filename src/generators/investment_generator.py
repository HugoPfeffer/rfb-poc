from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger, log_execution_time
from src.validation.validation_framework import (
    DataFrameValidationStrategy,
    validate_input,
    validate_config
)
from src.generators.components import (
    DataComponent,
    DataPersistence,
    DataTransformation,
    FraudScenarioGenerator
)
from src.generators.employment_generator import EmploymentDataGenerator

class InvestmentDataGenerator(DataComponent):
    """Generator for synthetic investment data with fraud scenarios."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the investment data generator."""
        super().__init__(config)
        self.persistence = DataPersistence()
        self.transformation = DataTransformation()
        self.fraud_generator = FraudScenarioGenerator(config)
        self.employment_generator = EmploymentDataGenerator(config)
        
        # Load settings from config
        if not hasattr(config_manager, '_config') or config_manager._config is None:
            config_manager.initialize()
            
        self.test_settings = config_manager.config.get('test_settings', {})
        if not self.test_settings:
            app_logger.warning("No test settings found in config, using defaults")
            self.test_settings = {
                'default_test_size': 100,
                'large_test_size': 1000,
                'delta_tolerance': 0.05,
                'asset_allocation': {
                    'atol': 1e-4,
                    'stocks_range': [0.4, 0.8],
                    'bonds_range': [0.1, 0.4],
                    'cash_range': [0.05, 0.2],
                    'real_estate_range': [0.0, 0.2]
                },
                'lifestyle_thresholds': {
                    'ratio_threshold': 0.4,
                    'luxury_spending_ratio': 0.25
                }
            }
        
        # Initialize fraud probabilities from config
        fraud_settings = self.config.get('fraud_scenarios', {})
        if not fraud_settings:
            raise ValueError("Required fraud_scenarios settings not found in config")
            
        self.fraud_probabilities = {}
        for fraud_type in ['salary_misreporting', 'suspicious_lifestyle', 'rapid_transactions']:
            prob = fraud_settings.get(fraud_type, {}).get('probability')
            if prob is None:
                raise ValueError(f"Required fraud_scenarios.{fraud_type}.probability not found in config")
            self.fraud_probabilities[fraud_type] = prob
            
        # Initialize validation strategy
        self.validator = DataFrameValidationStrategy(
            required_columns=[
                'industry', 'experience_level', 'salary', 'reported_salary',
                'annual_investment', 'stocks', 'bonds', 'cash', 'real_estate',
                'expected_annual_return', 'expected_value_1yr',
                'luxury_spending', 'travel_spending', 'lifestyle_ratio'
            ],
            column_types={
                'salary': np.float64,
                'reported_salary': np.float64,
                'annual_investment': np.float64,
                'stocks': np.float64,
                'bonds': np.float64,
                'cash': np.float64,
                'real_estate': np.float64,
                'expected_annual_return': np.float64,
                'expected_value_1yr': np.float64,
                'luxury_spending': np.float64,
                'travel_spending': np.float64,
                'lifestyle_ratio': np.float64,
                'is_fraudulent': bool,
                'fraud_type': pd.StringDtype(),
                'suspicious_lifestyle': bool
            }
        )
        
        # Investment parameters from config
        self.investment_rates = {}
        for level, settings in self.config['experience_levels'].items():
            if 'investment_rate' not in settings:
                raise ValueError(f"Required experience_levels.{level}.investment_rate not found in config")
            self.investment_rates[level] = tuple(settings['investment_rate'])
        
        # Asset allocation ranges from config
        asset_allocation = self.test_settings.get('asset_allocation')
        if not asset_allocation:
            raise ValueError("Required test_settings.asset_allocation not found in config")
            
        self.asset_classes = {
            'stocks': tuple(asset_allocation['stocks_range']),
            'bonds': tuple(asset_allocation['bonds_range']),
            'cash': tuple(asset_allocation['cash_range']),
            'real_estate': tuple(asset_allocation['real_estate_range'])
        }
        
        # Lifestyle indicators from config
        self.lifestyle_indicators = {
            'luxury_purchases': {},
            'travel_expenses': {}
        }
        for level, settings in self.config['experience_levels'].items():
            if 'luxury_spending' not in settings or 'travel_spending' not in settings:
                raise ValueError(f"Required luxury_spending and travel_spending not found in config for {level}")
            self.lifestyle_indicators['luxury_purchases'][level] = tuple(settings['luxury_spending'])
            self.lifestyle_indicators['travel_expenses'][level] = tuple(settings['travel_spending'])
    
    @log_execution_time(app_logger)
    @validate_config(['experience_levels', 'fraud_scenarios'])
    def generate(self, size: int) -> pd.DataFrame:
        """Generate synthetic investment dataset with fraud scenarios.
        
        Args:
            size: Number of records to generate
            
        Returns:
            pd.DataFrame: Generated investment dataset
        """
        app_logger.info(f"Generating investment dataset with {size} records")
        
        # Generate base employment data
        data = self.employment_generator.generate(size)
        
        # Add investment data
        data = self._add_portfolio_data(data)
        data = self._add_returns_data(data)
        data = self._add_lifestyle_indicators(data)
        
        # Add fraud scenarios
        data = self.fraud_generator.add_fraud_indicators(data)
        
        # Validate the generated data
        try:
            self.validator.validate(data)
            app_logger.info("Investment data validation successful")
        except Exception as e:
            app_logger.error(f"Investment data validation failed: {str(e)}")
            raise
        
        return data
    
    def _add_portfolio_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add portfolio information including investment amounts and allocations."""
        # Calculate investment amounts based on salary and experience level
        investment_amounts = []
        
        for _, row in data.iterrows():
            exp_level = row['experience_level']
            salary = row['salary']
            
            # Get investment rate range for experience level
            min_rate, max_rate = self.investment_rates[exp_level]
            
            # Calculate random investment rate within range
            investment_rate = np.random.uniform(min_rate, max_rate)
            investment_amount = salary * investment_rate
            investment_amounts.append(round(investment_amount, 2))
        
        data['annual_investment'] = pd.Series(investment_amounts, dtype='float64')
        
        # Generate asset allocations
        allocations = []
        for _ in range(len(data)):
            allocation = self._generate_allocation()
            allocations.append(allocation)
        
        # Add allocation columns
        allocation_df = pd.DataFrame(allocations)
        data = pd.concat([data, allocation_df], axis=1)
        
        return data
    
    def _generate_allocation(self) -> Dict[str, float]:
        """Generate random asset allocation that sums to exactly 1.0.
        
        Returns:
            Dict[str, float]: Asset allocation dictionary with keys as asset types
                             and values as allocation percentages.
        """
        MAX_ATTEMPTS = 10
        ATOL = 1e-4  # Absolute tolerance for floating point comparison
        
        for attempt in range(MAX_ATTEMPTS):
            try:
                allocation = {}
                remaining = 1.0
                assets = list(self.asset_classes.items())
                
                # Generate initial allocations for all assets
                for asset, (min_alloc, max_alloc) in assets:
                    if asset == assets[-1][0]:  # Last asset
                        alloc = remaining
                    else:
                        # Calculate maximum possible allocation considering remaining assets' minimums
                        remaining_min_sum = sum(min_a for _, (min_a, _) in assets[assets.index((asset, (min_alloc, max_alloc)))+1:])
                        max_possible = min(max_alloc, remaining - remaining_min_sum)
                        
                        if max_possible < min_alloc:
                            raise ValueError("Cannot satisfy minimum allocation")
                        
                        alloc = np.random.uniform(min_alloc, max_possible)
                    
                    allocation[asset] = round(alloc, 4)
                    remaining -= alloc
                
                # Verify all allocations are within bounds
                for asset, (min_alloc, max_alloc) in assets:
                    if not (min_alloc - ATOL <= allocation[asset] <= max_alloc + ATOL):
                        raise ValueError(f"Asset {asset} allocation outside bounds")
                
                # Normalize to ensure exact sum of 1.0
                total = sum(allocation.values())
                if not np.isclose(total, 1.0, atol=ATOL):
                    # Adjust the largest allocation to make sum exactly 1.0
                    largest_asset = max(allocation.items(), key=lambda x: x[1])[0]
                    allocation[largest_asset] = round(allocation[largest_asset] + (1.0 - total), 4)
                
                # Final verification
                if not np.isclose(sum(allocation.values()), 1.0, atol=ATOL):
                    raise ValueError("Failed to achieve exact sum of 1.0")
                
                return allocation
                
            except ValueError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise ValueError("Failed to generate valid asset allocation")
                continue
    
    def _add_returns_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add investment returns based on asset allocations."""
        # Get asset returns from config
        asset_returns = {}
        for asset, settings in self.config['asset_returns'].items():
            asset_returns[asset] = (settings['return'], settings['volatility'])
        
        # Calculate portfolio returns
        portfolio_returns = []
        
        for _, row in data.iterrows():
            total_return = 0
            for asset, (avg_return, volatility) in asset_returns.items():
                allocation = row[asset]
                # Generate random return with normal distribution
                asset_return = np.random.normal(avg_return, volatility)
                total_return += allocation * asset_return
            
            portfolio_returns.append(round(total_return, 4))
        
        data['expected_annual_return'] = pd.Series(portfolio_returns, dtype='float64')
        data['expected_value_1yr'] = round(
            data['annual_investment'] * (1 + data['expected_annual_return']), 
            2
        )
        
        return data
    
    def _add_lifestyle_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lifestyle indicators to help identify potential fraud."""
        size = len(data)
        luxury_spending = np.zeros(size)
        travel_spending = np.zeros(size)
        
        for idx, row in data.iterrows():
            exp_level = row['experience_level']
            salary = float(row['salary'])
            
            # Generate luxury purchases
            min_lux, max_lux = self.lifestyle_indicators['luxury_purchases'][exp_level]
            luxury_spending[idx] = round(salary * np.random.uniform(min_lux, max_lux), 2)
            
            # Generate travel expenses
            min_travel, max_travel = self.lifestyle_indicators['travel_expenses'][exp_level]
            travel_spending[idx] = round(salary * np.random.uniform(min_travel, max_travel), 2)
        
        # Add spending columns
        data['luxury_spending'] = pd.Series(luxury_spending, dtype='float64')
        data['travel_spending'] = pd.Series(travel_spending, dtype='float64')
        
        # Add lifestyle ratio indicators
        lifestyle_ratio = (data['luxury_spending'] + data['travel_spending']) / data['reported_salary']
        data['lifestyle_ratio'] = round(lifestyle_ratio, 4)
        
        # Add suspicious activity flags
        data['suspicious_lifestyle'] = (
            (data['lifestyle_ratio'] > 0.4) |  # Spending >40% of reported income
            (data['luxury_spending'] > data['reported_salary'] * 0.25)  # Luxury >25% of income
        )
        
        return data
    
    def save_dataset(self, data: pd.DataFrame, format: str = 'csv') -> Path:
        """Save the generated dataset.
        
        Args:
            data: DataFrame to save
            format: File format (csv, parquet, json)
            
        Returns:
            Path: Path to saved file
        """
        return self.persistence.save(data, 'investment_data', format) 