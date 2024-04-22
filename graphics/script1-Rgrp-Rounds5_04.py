import matplotlib.pyplot as plt
import numpy as np

categorias = [
    'WAVG Fair NR', 'WAVG NR', 
    'WAVG Fair Loss', 'WAVG Loss', 
    'WAVG Fair Rindv', 'WAVG Rindv',
    'AVG Fair',
    'AVG',
    'NFS'
]

valores = [
    0.253794730, 0.333889246,
    0.191555917, 0.346457183,
    0.176923722, 0.350863844, 
    (0.201010257 + 0.201281041 + 0.209656239)/3.0,
    (0.346341550 + 0.266793668 + 0.296403408)/3.0,
    (0.483585000 + 0.480675876 + 0.483972609)/3.0,
]

cores = [
    'lightgreen', 'green',
    'lightblue', 'blue', 
    'pink', 'red', 
    'lightgray',
    'gray',
    'black'
]

# cores = [
#     'palegreen', 'lightgreen', 'limegreen', 'forestgreen', 'darkgreen',
#     'lightblue', 'skyblue', 'blue', 'darkblue', 'navy',
#     'pink', 'lightcoral', 'red', 'darkred', 'maroon'
# ]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)
ax.set_xlabel('Injustiça do Grupo')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação e Agrupamentos (Rounds = 5)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(right=max(valores)*1.1)

# Adicionando linhas mais grossas para separar os grupos de 5
ax.axhline(2-0.5, color='black', linewidth=2)
ax.axhline(4-0.5, color='black', linewidth=2)
ax.axhline(6-0.5, color='black', linewidth=2)
ax.axhline(8-0.5, color='black', linewidth=2)

# Adicionando valores nas barras
for bar, value in zip(bars, valores):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()
plt.show()
