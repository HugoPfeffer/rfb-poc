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
        
        # Check config path if provided
        if config and 'config_path' in config:
            config_path = Path(config['config_path'])
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.industry_data = self._load_industry_ranges()
        self.experience_levels = ['entry', 'mid', 'senior']
        
    def _load_industry_ranges(self) -> List[Dict[str, Any]]:
        """Load industry salary ranges from configuration file."""
        config_path = Path('data/configs/industry_ranges.json')
        if not config_path.exists():
            raise FileNotFoundError("Industry ranges configuration file not found")
            
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate(self, size: int) -> pd.DataFrame:
        """Generate synthetic employment dataset.
        
        Args:
            size: Number of employment records to generate
            
        Returns:
            pd.DataFrame: Generated employment dataset
        """
        # Generate base data
        self.data = pd.DataFrame()
        
        # Generate industry and experience level distributions
        self.add_industry_data(size)
        self.add_experience_levels(size)
        self.add_salary_data()
        
        return self.data
    
    def add_industry_data(self, size: int) -> None:
        """Add industry data to the dataset.
        
        Args:
            size: Number of records to generate
        """
        # Reset/initialize DataFrame with correct size
        self.data = pd.DataFrame(index=range(size))
        
        industries = [ind['industry'] for ind in self.industry_data]
        self.data['industry'] = np.random.choice(industries, size=size)
    
    def add_experience_levels(self, size: int) -> None:
        """Add experience levels with specified distribution.
        
        Args:
            size: Number of records to generate
        """
        # Ensure data DataFrame exists
        if self.data is None or len(self.data) != size:
            self.data = pd.DataFrame(index=range(size))
        
        # Generate experience levels with specified probabilities
        self.data['experience_level'] = np.random.choice(
            self.experience_levels,
            size=size,
            p=[0.4, 0.4, 0.2]  # 40% entry, 40% mid, 20% senior
        )
    
    def add_salary_data(self) -> None:
        """Add salary information to the dataset."""
        salaries = []
        
        for _, row in self.data.iterrows():
            industry_info = next(
                ind for ind in self.industry_data 
                if ind['industry'] == row['industry']
            )
            base_salary = industry_info['salary_ranges'][row['experience_level']]
            
            # Add some random variation (±10%)
            variation = np.random.uniform(-0.1, 0.1)
            salary = int(base_salary * (1 + variation))
            salaries.append(salary)
            
        self.data['salary'] = salaries
    
    def validate(self) -> bool:
        """Validate the generated employment dataset.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if self.data is None:
            return False
            
        required_columns = {'industry', 'experience_level', 'salary'}
        if not all(col in self.data.columns for col in required_columns):
            return False
            
        if not all(self.data['experience_level'].isin(self.experience_levels)):
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

    