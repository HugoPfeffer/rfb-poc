from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

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

class EmploymentDataGenerator(DataComponent):
    """Generator for synthetic employment data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the employment data generator.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.persistence = DataPersistence()
        self.transformation = DataTransformation()
        self.fraud_generator = FraudScenarioGenerator(config)
        
        # Load industry data
        self.industry_data = self._load_industry_ranges()
        self.experience_levels = list(self.config['experience_levels'].keys())
        
        # Initialize validation strategy
        self.validator = DataFrameValidationStrategy(
            required_columns=['industry', 'experience_level', 'salary', 'reported_salary'],
            column_types={
                'salary': np.float64,
                'reported_salary': np.float64,
                'is_fraudulent': bool,
                'fraud_type': pd.StringDtype()
            }
        )
    
    def _load_industry_ranges(self) -> List[Dict[str, Any]]:
        """Load industry salary ranges from configuration file."""
        industry_file = Path(__file__).parent.parent.parent / 'data' / 'configs' / 'industry_ranges.json'
        if not industry_file.exists():
            raise FileNotFoundError(f"Industry ranges file not found: {industry_file}")
            
        with open(industry_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or 'industry_data' not in data:
                raise ValueError("Invalid industry ranges file format")
            return data['industry_data']
    
    def add_industry_data(self, size: int) -> None:
        """Add industry data to the dataset.
        
        Args:
            size: Number of records to generate
        """
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame(index=range(size))
            
        industries = [ind['industry'] for ind in self.industry_data]
        self.data['industry'] = self.rng.choice(industries, size=size)
    
    def add_experience_levels(self, size: int) -> None:
        """Add experience levels to the dataset.
        
        Args:
            size: Number of records to generate
        """
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame(index=range(size))
            
        probabilities = [
            self.config['experience_levels'][level]['probability']
            for level in self.experience_levels
        ]
        self.data['experience_level'] = self.rng.choice(
            self.experience_levels,
            size=size,
            p=probabilities
        )
    
    @log_execution_time(app_logger)
    @validate_config(['experience_levels', 'salary_distribution'])
    def generate(self, size: int) -> pd.DataFrame:
        """Generate synthetic employment dataset.
        
        Args:
            size: Number of records to generate
            
        Returns:
            pd.DataFrame: Generated employment dataset
        """
        app_logger.info(f"Generating employment dataset with {size} records")
        
        try:
            # Initialize data
            self.data = pd.DataFrame(index=range(size))
            
            # Add industry data
            industries = [ind['industry'] for ind in self.industry_data]
            self.data['industry'] = self.rng.choice(industries, size=size)
            
            # Add experience levels
            probabilities = [
                self.config['experience_levels'][level]['probability']
                for level in self.experience_levels
            ]
            self.data['experience_level'] = self.rng.choice(
                self.experience_levels,
                size=size,
                p=probabilities
            )
            
            # Add salary data
            self.data = self._add_salary_data(self.data)
            
            # Add fraud scenarios
            self.data = self.fraud_generator.add_fraud_indicators(self.data)
            
            # Validate the generated data
            self.validator.validate(self.data)
            app_logger.info("Employment data validation successful")
            
            return self.data.copy()
            
        except Exception as e:
            app_logger.error(f"Error in generate: {str(e)}")
            raise
    
    def _add_salary_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add salary information to the dataset."""
        salaries = []
        
        # Get salary distribution settings
        dist_settings = self.config['salary_distribution']
        
        # Define ranges and probabilities
        ranges = ['normal_range', 'low_outlier', 'high_outlier']
        probabilities = [
            dist_settings[r]['probability']
            for r in ranges
        ]
        
        for _, row in data.iterrows():
            industry_info = next(
                ind for ind in self.industry_data 
                if ind['industry'] == row['industry']
            )
            # Parse salary range string into min and max values
            salary_range = industry_info['salary_ranges'][row['experience_level']]
            min_salary, max_salary = map(float, salary_range.split('-'))
            
            # Generate base salary within the range
            base_salary = self.rng.uniform(min_salary, max_salary)
            
            # Select which range to use
            range_type = self.rng.choice(ranges, p=probabilities)
            range_settings = dist_settings[range_type]
            
            # Generate ratio based on the selected range
            ratio = self.rng.uniform(
                range_settings['min_ratio'],
                range_settings['max_ratio']
            )
            
            # Calculate and round the salary
            salary = round(base_salary * ratio, 2)
            salaries.append(salary)
        
        data['salary'] = pd.Series(salaries, dtype='float64')
        data['reported_salary'] = data['salary'].copy()
        
        return data
    
    def save_industry_dataset(self) -> Path:
        """Save the generated dataset.
        
        Returns:
            Path: Path to saved file
        """
        if not hasattr(self, 'data'):
            raise ValueError("No data to save. Generate data first.")
            
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'industry_dataset_{timestamp}'
        return self.persistence.save(self.data, filename, format='csv') 