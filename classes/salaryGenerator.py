# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
class SalaryGenerator:
    def __init__(self):
        # Initialize sectors data
        self.sectors_data = [
            {
                'Setor': "Tecnologia da Informação",
                'Entry_Min': 50000, 'Entry_Max': 70000,
                'Mid_Min': 84000, 'Mid_Max': 105000,
                'Senior_Min': 136500, 'Senior_Max': 210000
            },
            {
                'Setor': "Saúde",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Educação",
                'Entry_Min': 30000, 'Entry_Max': 42000,
                'Mid_Min': 50400, 'Mid_Max': 63000,
                'Senior_Min': 81900, 'Senior_Max': 126000
            },
            {
                'Setor': "Finanças",
                'Entry_Min': 50000, 'Entry_Max': 70000,
                'Mid_Min': 84000, 'Mid_Max': 105000,
                'Senior_Min': 136500, 'Senior_Max': 210000
            },
            {
                'Setor': "Varejo",
                'Entry_Min': 25000, 'Entry_Max': 35000,
                'Mid_Min': 42000, 'Mid_Max': 52500,
                'Senior_Min': 68250, 'Senior_Max': 105000
            },
            {
                'Setor': "Indústria Manufatureira",
                'Entry_Min': 30000, 'Entry_Max': 42000,
                'Mid_Min': 50400, 'Mid_Max': 63000,
                'Senior_Min': 81900, 'Senior_Max': 126000
            },
            {
                'Setor': "Construção Civil",
                'Entry_Min': 35000, 'Entry_Max': 49000,
                'Mid_Min': 58800, 'Mid_Max': 73500,
                'Senior_Min': 95550, 'Senior_Max': 147000
            },
            {
                'Setor': "Agricultura",
                'Entry_Min': 20000, 'Entry_Max': 28000,
                'Mid_Min': 33600, 'Mid_Max': 42000,
                'Senior_Min': 54600, 'Senior_Max': 84000
            },
            {
                'Setor': "Transporte e Logística",
                'Entry_Min': 25000, 'Entry_Max': 35000,
                'Mid_Min': 42000, 'Mid_Max': 52500,
                'Senior_Min': 68250, 'Senior_Max': 105000
            },
            {
                'Setor': "Telecomunicações",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Energia",
                'Entry_Min': 45000, 'Entry_Max': 63000,
                'Mid_Min': 75600, 'Mid_Max': 94500,
                'Senior_Min': 122850, 'Senior_Max': 189000
            },
            {
                'Setor': "Mídia e Entretenimento",
                'Entry_Min': 30000, 'Entry_Max': 42000,
                'Mid_Min': 50400, 'Mid_Max': 63000,
                'Senior_Min': 81900, 'Senior_Max': 126000
            },
            {
                'Setor': "Hotelaria e Turismo",
                'Entry_Min': 20000, 'Entry_Max': 28000,
                'Mid_Min': 33600, 'Mid_Max': 42000,
                'Senior_Min': 54600, 'Senior_Max': 84000
            },
            {
                'Setor': "Serviços Jurídicos",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Recursos Humanos",
                'Entry_Min': 30000, 'Entry_Max': 42000,
                'Mid_Min': 50400, 'Mid_Max': 63000,
                'Senior_Min': 81900, 'Senior_Max': 126000
            },
            {
                'Setor': "Marketing e Publicidade",
                'Entry_Min': 30000, 'Entry_Max': 42000,
                'Mid_Min': 50400, 'Mid_Max': 63000,
                'Senior_Min': 81900, 'Senior_Max': 126000
            },
            {
                'Setor': "Pesquisa e Desenvolvimento",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Farmacêutica",
                'Entry_Min': 45000, 'Entry_Max': 63000,
                'Mid_Min': 75600, 'Mid_Max': 94500,
                'Senior_Min': 122850, 'Senior_Max': 189000
            },
            {
                'Setor': "Automotivo",
                'Entry_Min': 35000, 'Entry_Max': 49000,
                'Mid_Min': 58800, 'Mid_Max': 73500,
                'Senior_Min': 95550, 'Senior_Max': 147000
            },
            {
                'Setor': "Alimentação e Bebidas",
                'Entry_Min': 25000, 'Entry_Max': 35000,
                'Mid_Min': 42000, 'Mid_Max': 52500,
                'Senior_Min': 68250, 'Senior_Max': 105000
            },
            {
                'Setor': "Consultoria Empresarial",
                'Entry_Min': 50000, 'Entry_Max': 70000,
                'Mid_Min': 84000, 'Mid_Max': 105000,
                'Senior_Min': 136500, 'Senior_Max': 210000
            },
            {
                'Setor': "Seguros",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Imobiliário",
                'Entry_Min': 35000, 'Entry_Max': 49000,
                'Mid_Min': 58800, 'Mid_Max': 73500,
                'Senior_Min': 95550, 'Senior_Max': 147000
            },
            {
                'Setor': "Mineração",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Petróleo e Gás",
                'Entry_Min': 50000, 'Entry_Max': 70000,
                'Mid_Min': 84000, 'Mid_Max': 105000,
                'Senior_Min': 136500, 'Senior_Max': 210000
            },
            {
                'Setor': "Serviços Ambientais",
                'Entry_Min': 30000, 'Entry_Max': 42000,
                'Mid_Min': 50400, 'Mid_Max': 63000,
                'Senior_Min': 81900, 'Senior_Max': 126000
            },
            {
                'Setor': "Aeroespacial",
                'Entry_Min': 45000, 'Entry_Max': 63000,
                'Mid_Min': 75600, 'Mid_Max': 94500,
                'Senior_Min': 122850, 'Senior_Max': 189000
            },
            {
                'Setor': "Biotecnologia",
                'Entry_Min': 40000, 'Entry_Max': 56000,
                'Mid_Min': 67200, 'Mid_Max': 84000,
                'Senior_Min': 109200, 'Senior_Max': 168000
            },
            {
                'Setor': "Comércio Exterior",
                'Entry_Min': 35000, 'Entry_Max': 49000,
                'Mid_Min': 58800, 'Mid_Max': 73500,
                'Senior_Min': 95550, 'Senior_Max': 147000
            },
            {
                'Setor': "Serviços Financeiros",
                'Entry_Min': 50000, 'Entry_Max': 70000,
                'Mid_Min': 84000, 'Mid_Max': 105000,
                'Senior_Min': 136500, 'Senior_Max': 210000
            },
            {
                'Setor': "Administração Pública",
                'Entry_Min': 25000, 'Entry_Max': 35000,
                'Mid_Min': 42000, 'Mid_Max': 52500,
                'Senior_Min': 68250, 'Senior_Max': 105000
            },
            {
                'Setor': "Organizações Sem Fins Lucrativos",
                'Entry_Min': 20000, 'Entry_Max': 28000,
                'Mid_Min': 33600, 'Mid_Max': 42000,
                'Senior_Min': 54600, 'Senior_Max': 84000
            },
            {
                'Setor': "Esportes e Recreação",
                'Entry_Min': 20000, 'Entry_Max': 28000,
                'Mid_Min': 33600, 'Mid_Max': 42000,
                'Senior_Min': 54600, 'Senior_Max': 84000
            },
            {
                'Setor': "Artes e Cultura",
                'Entry_Min': 20000, 'Entry_Max': 28000,
                'Mid_Min': 33600, 'Mid_Max': 42000,
                'Senior_Min': 54600, 'Senior_Max': 84000
            },
            {
                'Setor': "Segurança e Vigilância",
                'Entry_Min': 25000, 'Entry_Max': 35000,
                'Mid_Min': 42000, 'Mid_Max': 52500,
                'Senior_Min': 68250, 'Senior_Max': 105000
            }
        ]
        self.sectors_df = pd.DataFrame(self.sectors_data)

    def generate_salaries(self, num_samples=10):
        """
        Generate salary data for all sectors and experience levels
        
        Args:
            num_samples (int): Number of samples per sector and level
            
        Returns:
            pd.DataFrame: DataFrame containing generated salary data
        """
        data = []
        for _, row in self.sectors_df.iterrows():
            entry_level = np.random.randint(row['Entry_Min'], row['Entry_Max'], num_samples)
            mid_career = np.random.randint(row['Mid_Min'], row['Mid_Max'], num_samples)
            senior_level = np.random.randint(row['Senior_Min'], row['Senior_Max'], num_samples)

            for i in range(num_samples):
                data.append({"Sector": row['Setor'], "Level": "Entry", "Salary": entry_level[i]})
                data.append({"Sector": row['Setor'], "Level": "Mid", "Salary": mid_career[i]})
                data.append({"Sector": row['Setor'], "Level": "Senior", "Salary": senior_level[i]})
        return pd.DataFrame(data)

    def generate_sample_dataset(self, num_samples_per_sector=100):
        """
        Generate a sample salary dataset using default parameters
        
        Args:
            num_samples_per_sector (int): Number of samples to generate per sector (default: 100)
            
        Returns:
            pd.DataFrame: DataFrame containing generated salary data
        """
        return self.generate_salaries(num_samples_per_sector)



