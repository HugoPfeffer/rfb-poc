# %%
from faker import Faker
from providers.industriaProvider import industriaProvider
import pandas as pd
import random
import numpy as np
from scipy.stats.mstats import winsorize



# %%
fake = Faker('pt_BR')

# %%
fake.add_provider(industriaProvider)

# %%
def create_rows(num=1):
    output = [{"name":fake.name(),
                "Endereço":fake.address(),
                "Nome":fake.name(),
                "CPF":fake.cpf(),
                # "Setor de Trabalho":(setor := fake.industria()),
                "Faixa Salarial":fake.range_salarial(),
                "Nascimento":fake.date_of_birth(minimum_age=18, maximum_age=65).strftime('%d/%m/%Y'),
                "Placa":fake.license_plate(),
                "RENAVAM":fake.vin(),
                "randomdata":random.randint(1000,2000)} for x in range(num)]
    return output

# %%
create_rows(10)


