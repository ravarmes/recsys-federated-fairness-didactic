import torch
import pandas as pd

# Criando um tensor de exemplo
tensor_exemplo = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Convertendo o tensor para um array NumPy
array_numpy = tensor_exemplo.numpy()

# Criando um DataFrame do pandas a partir do array NumPy
df = pd.DataFrame(array_numpy)

print(df)