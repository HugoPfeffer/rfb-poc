# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Define base salary ranges for each sector
sectors = {
    "Tecnologia da Informação": [50000, 70000, 84000, 105000, 136500, 210000],
    "Saúde": [40000, 56000, 67200, 84000, 109200, 168000],
    "Educação": [30000, 42000, 50400, 63000, 81900, 126000],
    "Finanças": [50000, 70000, 84000, 105000, 136500, 210000],
    "Varejo": [25000, 35000, 42000, 52500, 68250, 105000],
    "Indústria Manufatureira": [30000, 42000, 50400, 63000, 81900, 126000],
    "Construção Civil": [35000, 49000, 58800, 73500, 95550, 147000],
    "Agricultura": [20000, 28000, 33600, 42000, 54600, 84000],
    "Transporte e Logística": [25000, 35000, 42000, 52500, 68250, 105000],
    "Telecomunicações": [40000, 56000, 67200, 84000, 109200, 168000],
    "Energia": [45000, 63000, 75600, 94500, 122850, 189000],
    "Mídia e Entretenimento": [30000, 42000, 50400, 63000, 81900, 126000],
    "Hotelaria e Turismo": [20000, 28000, 33600, 42000, 54600, 84000],
    "Serviços Jurídicos": [40000, 56000, 67200, 84000, 109200, 168000],
    "Recursos Humanos": [30000, 42000, 50400, 63000, 81900, 126000],
    "Marketing e Publicidade": [30000, 42000, 50400, 63000, 81900, 126000],
    "Pesquisa e Desenvolvimento": [40000, 56000, 67200, 84000, 109200, 168000],
    "Farmacêutica": [45000, 63000, 75600, 94500, 122850, 189000],
    "Automotivo": [35000, 49000, 58800, 73500, 95550, 147000],
    "Alimentação e Bebidas": [25000, 35000, 42000, 52500, 68250, 105000],
    "Consultoria Empresarial": [50000, 70000, 84000, 105000, 136500, 210000],
    "Seguros": [40000, 56000, 67200, 84000, 109200, 168000],
    "Imobiliário": [35000, 49000, 58800, 73500, 95550, 147000],
    "Mineração": [40000, 56000, 67200, 84000, 109200, 168000],
    "Petróleo e Gás": [50000, 70000, 84000, 105000, 136500, 210000],
    "Serviços Ambientais": [30000, 42000, 50400, 63000, 81900, 126000],
    "Aeroespacial": [45000, 63000, 75600, 94500, 122850, 189000],
    "Biotecnologia": [40000, 56000, 67200, 84000, 109200, 168000],
    "Comércio Exterior": [35000, 49000, 58800, 73500, 95550, 147000],
    "Serviços Financeiros": [50000, 70000, 84000, 105000, 136500, 210000],
    "Administração Pública": [25000, 35000, 42000, 52500, 68250, 105000],
    "Organizações Sem Fins Lucrativos": [20000, 28000, 33600, 42000, 54600, 84000],
    "Esportes e Recreação": [20000, 28000, 33600, 42000, 54600, 84000],
    "Artes e Cultura": [20000, 28000, 33600, 42000, 54600, 84000],
    "Segurança e Vigilância": [25000, 35000, 42000, 52500, 68250, 105000],
}

# %%
# Function to generate salaries for each sector
def generate_salaries(sector_ranges, num_samples):
    data = []
    for sector, ranges in sector_ranges.items():
        entry_level = np.random.randint(ranges[0], ranges[1], num_samples)
        mid_career = np.random.randint(ranges[2], ranges[3], num_samples)
        senior_level = np.random.randint(ranges[4], ranges[5], num_samples)

        for i in range(num_samples):
            data.append({"Sector": sector, "Level": "Entry", "Salary": entry_level[i]})
            data.append({"Sector": sector, "Level": "Mid", "Salary": mid_career[i]})
            data.append({"Sector": sector, "Level": "Senior", "Salary": senior_level[i]})
    return pd.DataFrame(data)

# %%
# Generate dataset
num_samples_per_sector = 100
dataset = generate_salaries(sectors, num_samples_per_sector)



