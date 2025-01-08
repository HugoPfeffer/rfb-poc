from faker.providers import BaseProvider
import random
import numpy as np
from classes.salaryGenerator import SalaryGenerator
from classes.investimentsGenerator import InvestmentsGenerator

class industriaProvider(BaseProvider):
    salary_gen = SalaryGenerator()  # Class-level instance

    def __init__(self, generator):
        super().__init__(generator)
        self.salary_df = self.salary_gen.generate_salaries()
        # Get unique sectors from the salary generator
        self.setores = self.salary_gen.sectors_df['Setor'].unique()

    def industria(self):
        """Returns a random industry sector"""
        return random.choice(self.setores)

    def range_salarial(self, setor: str = None, nivel: str = None) -> int:
        """
        Returns a salary based on sector and level using the salary dataframe.
        
        Args:
            setor (str, optional): Industry sector. If None, a random sector is chosen.
            nivel (str, optional): Career level ('Entry', 'Mid', or 'Senior'). If None, a random level is chosen.
            
        Returns:
            int: A salary value within the specified range
        """
        if setor is None:
            setor = random.choice(self.salary_df['setor'].unique())
        
        # Define possible levels and their mappings
        nivel_map = {
            'entry': 'junior',
            'mid': 'pleno',
            'senior': 'senior'
        }
        
        # If nivel is None, randomly choose one
        if nivel is None:
            nivel = random.choice(['entry', 'mid', 'senior'])
        
        nivel = nivel_map.get(nivel.lower(), 'junior')
        
        salary_range = self.salary_df[
            (self.salary_df['setor'] == setor) & 
            (self.salary_df['nivel'] == nivel)
        ]
        
        if salary_range.empty:
            return 0
            
        min_salary = salary_range['min_salary'].iloc[0]
        max_salary = salary_range['max_salary'].iloc[0]
        
        return np.random.randint(min_salary, max_salary)