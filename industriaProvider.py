from faker.providers import BaseProvider
import random
import numpy as np

class industriaProvider(BaseProvider):
    setores = {
        "tecnologia": "Tecnologia da Informação",
        "saude": "Saúde",
        "educacao": "Educação",
        "financas": "Finanças",
        "varejo": "Varejo",
        "manufatura": "Indústria Manufatureira",
        "construcao": "Construção Civil",
        "agricultura": "Agricultura",
        "logistica": "Transporte e Logística",
        "telecom": "Telecomunicações",
        "energia": "Energia",
        "midia": "Mídia e Entretenimento",
        "turismo": "Hotelaria e Turismo",
        "juridico": "Serviços Jurídicos",
        "rh": "Recursos Humanos",
        "marketing": "Marketing e Publicidade",
        "pesquisa": "Pesquisa e Desenvolvimento",
        "farmaceutica": "Farmacêutica",
        "automotivo": "Automotivo",
        "alimentos": "Alimentação e Bebidas",
        "consultoria": "Consultoria Empresarial",
        "seguros": "Seguros",
        "imobiliario": "Imobiliário",
        "mineracao": "Mineração",
        "petroleo": "Petróleo e Gás",
        "ambiental": "Serviços Ambientais",
        "aeroespacial": "Aeroespacial",
        "biotecnologia": "Biotecnologia",
        "comercio_exterior": "Comércio Exterior",
        "servicos_financeiros": "Serviços Financeiros",
        "setor_publico": "Administração Pública",
        "ong": "Organizações Sem Fins Lucrativos",
        "esportes": "Esportes e Recreação",
        "cultura": "Artes e Cultura",
        "seguranca": "Segurança e Vigilância"
    }

    faixas_salariais = {
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

    def industria(self) -> str:
        return random.choice(list(self.setores.values()))

    def range_salarial(self, setor: str = None, nivel: str = "Entry") -> int:
        """
        Returns a salary based on sector and level.
        
        Args:
            setor (str, optional): Industry sector. If None, a random sector is chosen.
            nivel (str, optional): Career level ('Entry', 'Mid', or 'Senior'). Defaults to 'Entry'.
            
        Returns:
            int: A salary value within the specified range
        """
        if setor is None:
            setor = random.choice(list(self.faixas_salariais.keys()))
        
        if setor not in self.faixas_salariais:
            return 0
            
        ranges = self.faixas_salariais[setor]
        
        if nivel.lower() == "entry":
            return np.random.randint(ranges[0], ranges[1])
        elif nivel.lower() == "mid":
            return np.random.randint(ranges[2], ranges[3])
        elif nivel.lower() == "senior":
            return np.random.randint(ranges[4], ranges[5])
        else:
            return np.random.randint(ranges[0], ranges[1])  # Default to entry level