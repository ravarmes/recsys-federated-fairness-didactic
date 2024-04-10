import torch
import pandas as pd

# Criando um DataFrame de exemplo
df_exemplo = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Convertendo o DataFrame para um array NumPy
array_numpy = df_exemplo.values

# Criando um tensor PyTorch a partir do array NumPy
tensor = torch.tensor(array_numpy)

print("DataFrame")
print(df_exemplo)

print("\n")

print("Tensor")
print(tensor)
