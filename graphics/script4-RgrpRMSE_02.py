import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = [
    'WAVG Fair NR', 'WAVG NR', 
    'WAVG Fair Loss', 'WAVG Loss', 
    'WAVG Fair Rindv', 'WAVG Rindv',
    'AVG Fair',
    'AVG',
    'NFS'
]

valores = [
    [0.253794730], [0.333889246], 
    [0.191555917], [0.346457183], 
    [0.176923722], [0.350863844], 
    [(0.201010257 + 0.201281041 + 0.209656239) / 3], 
    [(0.346341550 + 0.266793668 + 0.296403408) / 3], 
    [(0.483585000 + 0.480675876 + 0.483972609) / 3]
]

# Criando boxplot para as categorias
plt.figure(figsize=(10, 6))
plt.boxplot(valores, positions=range(1, len(categorias) + 1), patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.xticks(range(1, len(categorias) + 1), categorias, rotation=45, ha='right')
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Boxplot para Diferentes Categorias')
plt.tight_layout()
plt.show()
