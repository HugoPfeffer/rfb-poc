# %%
from faker import Faker
import pandas as pd
import random
from industriaProvider import industriaProvider

# %%
fake = Faker('pt_BR')

# %%
fake.add_provider(industriaProvider)
# setor = fake.industria()
# salario = fake.range_salarial(setor)
# print(setor)
# print(salario)

# %%
def create_rows(num=1):
    output = [{"name":fake.name(),
                "Endereço":fake.address(),
                "Nome":fake.name(),
                "CPF":fake.cpf(),
                "Setor de Trabalho":(setor := fake.industria()),
                "Faixa Salarial":fake.range_salarial(setor),
                "Nascimento":fake.date_of_birth(minimum_age=18, maximum_age=65).strftime('%d/%m/%Y'),
                "Placa":fake.license_plate(),
                "RENAVAM":fake.vin(),
                "randomdata":random.randint(1000,2000)} for x in range(num)]
    return output

# %%
create_rows(10)



# %%
