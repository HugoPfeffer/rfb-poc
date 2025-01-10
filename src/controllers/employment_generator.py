from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from .dataset_generator import DatasetGenerator

class EmploymentGenerator(DatasetGenerator):
    """Generator for synthetic employment data.
    
    This class generates synthetic employment data including salaries,
    job titles, experience levels, and other employment-related information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the employment generator.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        # Initialize data as empty DataFrame
        self.data = pd.DataFrame()
        
        # Load settings
        self.settings = self._load_settings()
        
        # Use numpy's random state initialized by parent class
        self.rng = np.random.RandomState(self.settings['random_seed'])
        
        # Load industry data
        self.industry_data = self._load_industry_ranges()
        
        # Get experience levels from settings
        self.experience_levels = list(self.settings['experience_levels'].keys())
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from configuration file."""
        config_path = Path('data/configs/settings.json')
        if not config_path.exists():
            raise FileNotFoundError("Settings file not found")
            
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate(self, size: int) -> pd.DataFrame:
        """Generate synthetic employment dataset.
        
        Args:
            size: Number of records to generate
            
        Returns:
            pd.DataFrame: Generated employment dataset
        """
        # Generate base data
        self.data = pd.DataFrame()
        
        # Generate industry and experience level distributions
        self.add_industry_data(size)
        self.add_experience_levels(size)
        self.add_salary_data()
        
        # Add fraud scenarios
        self.add_fraud_scenarios()
        
        return self.data
    
    def add_industry_data(self, size: int) -> None:
        """Add industry data to the dataset."""
        # Reset/initialize DataFrame with correct size
        self.data = pd.DataFrame(index=range(size))
        
        industries = [ind['industry'] for ind in self.industry_data]
        self.data['industry'] = self.rng.choice(industries, size=size)
    
    def add_experience_levels(self, size: int) -> None:
        """Add experience levels with specified distribution."""
        # Get probabilities from settings
        probabilities = [
            self.settings['experience_levels'][level]['probability']
            for level in self.experience_levels
        ]
        
        self.data['experience_level'] = self.rng.choice(
            self.experience_levels,
            size=size,
            p=probabilities
        )
    
    def add_salary_data(self) -> None:
        """Add salary information to the dataset."""
        salaries = []
        
        # Get salary distribution settings
        dist_settings = self.settings['salary_distribution']
        
        # Define ranges and probabilities
        ranges = ['normal_range', 'low_outlier', 'high_outlier']
        probabilities = [
            dist_settings[r]['probability']
            for r in ranges
        ]
        
        for _, row in self.data.iterrows():
            industry_info = next(
                ind for ind in self.industry_data 
                if ind['industry'] == row['industry']
            )
            base_salary = float(industry_info['salary_ranges'][row['experience_level']])
            
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
            
        self.data['salary'] = pd.Series(salaries, dtype='float64')
    
    def add_fraud_scenarios(self) -> None:
        """Add fraud scenarios to employment data."""
        if self.data is not None:
            size = len(self.data)
            fraud_settings = self.settings['fraud_scenarios']
            
            # Initialize fraud columns
            self.data['is_fraudulent'] = False
            self.data['fraud_type'] = 'none'
            self.data['reported_salary'] = self.data['salary'].astype('float64')
            
            # Add fraud scenarios based on probability from settings
            for idx in range(size):
                if self.rng.random() < fraud_settings['probability']:
                    self.data.loc[idx, 'is_fraudulent'] = True
                    
                    # Determine fraud type based on probability
                    if self.rng.random() < fraud_settings['salary_misreporting']['probability']:
                        # Salary misreporting
                        true_salary = float(self.data.loc[idx, 'salary'])
                        misreport_settings = fraud_settings['salary_misreporting']
                        reported_ratio = self.rng.uniform(
                            misreport_settings['min_ratio'],
                            misreport_settings['max_ratio']
                        )
                        reported_salary = true_salary * reported_ratio
                        self.data.loc[idx, 'reported_salary'] = round(reported_salary, 2)
                        self.data.loc[idx, 'fraud_type'] = 'salary_misreporting'
                    else:
                        # Experience level inflation
                        current_level = self.data.loc[idx, 'experience_level']
                        if current_level == 'entry':
                            self.data.loc[idx, 'experience_level'] = 'mid'
                            self.data.loc[idx, 'fraud_type'] = 'experience_inflation'
    
    def _load_industry_ranges(self) -> List[Dict[str, Any]]:
        """Load industry salary ranges from configuration file."""
        config_path = Path('data/configs/industry_ranges.json')
        if not config_path.exists():
            raise FileNotFoundError("Industry ranges configuration file not found")
            
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def validate(self) -> bool:
        """Validate the generated employment dataset.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if self.data is None:
            return False
            
        # Check required columns
        required_columns = {
            'industry', 'experience_level', 'salary', 
            'is_fraudulent', 'fraud_type', 'reported_salary'
        }
        if not all(col in self.data.columns for col in required_columns):
            return False
            
        # Validate experience levels
        if not all(self.data['experience_level'].isin(self.experience_levels)):
            return False
            
        # Validate industries
        valid_industries = {ind['industry'] for ind in self.industry_data}
        if not all(self.data['industry'].isin(valid_industries)):
            return False
            
        # Validate fraud indicators
        if not all(self.data['is_fraudulent'].isin([True, False])):
            return False
            
        if not all(self.data['fraud_type'].isin(['none', 'salary_misreporting', 'experience_inflation'])):
            return False
            
        # Validate salary relationships
        if not all(self.data['reported_salary'] <= self.data['salary']):
            return False
            
        return True
    
    def save(self, path: str, format: str = 'csv') -> None:
        """Save the generated dataset to disk.
        
        Args:
            path: Path where to save the dataset
            format: Format to save the data in (csv, parquet, etc.)
            
        Raises:
            ValueError: If data hasn't been generated or format is unsupported
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data has been generated yet")
            
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            self.data.to_csv(path, index=False)
        elif format.lower() == 'parquet':
            self.data.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def save_industry_dataset(self) -> str:
        """Save the industry dataset with a date-stamped filename.
        
        Returns:
            str: Path of the saved file
        
        Raises:
            ValueError: If data hasn't been generated yet
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data has been generated yet")
            
        # Create the clean data directory if it doesn't exist
        output_dir = Path('./data/clean')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with current date
        current_date = datetime.now().strftime('%Y%m%d')
        filename = f'industry_dataset_{current_date}.csv'
        
        # Construct full path
        output_path = output_dir / filename
        
        # Save the dataset
        self.data.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def add_job_titles(self) -> None:
        """Add job titles to the dataset."""
        # TODO: Implement job title generation logic
        pass
    
    def add_fraud_scenarios(self) -> None:
        """Add fraud scenarios to employment data."""
        if self.data is not None:
            size = len(self.data)
            
            # Initialize fraud columns
            self.data['is_fraudulent'] = False
            self.data['fraud_type'] = 'none'
            self.data['reported_salary'] = self.data['salary'].astype('float64')
            
            # Add fraud scenarios (10% chance for each record)
            for idx in range(size):
                if np.random.random() < 0.1:  # 10% fraud rate
                    self.data.loc[idx, 'is_fraudulent'] = True
                    
                    # 50-50 chance between salary misreporting and experience inflation
                    if np.random.random() < 0.5:
                        # Salary misreporting (report 70-90% of actual salary)
                        true_salary = float(self.data.loc[idx, 'salary'])
                        reported_salary = true_salary * np.random.uniform(0.7, 0.9)
                        self.data.loc[idx, 'reported_salary'] = round(reported_salary, 2)
                        self.data.loc[idx, 'fraud_type'] = 'salary_misreporting'
                    else:
                        # Experience level inflation
                        current_level = self.data.loc[idx, 'experience_level']
                        if current_level == 'entry':
                            self.data.loc[idx, 'experience_level'] = 'mid'
                            self.data.loc[idx, 'fraud_type'] = 'experience_inflation'

    