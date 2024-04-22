import matplotlib.pyplot as plt
import numpy as np

# Dados, agora listados de baixo para cima para coincidir com a ordem de cima para baixo no gráfico
categorias = [
    'WAVG Fair NR', 'AVG Fair NR', 'WAVG NR', 'AVG NR', 'NFS NR',
    'WAVG Fair Loss', 'AVG Fair Loss', 'WAVG Loss', 'AVG Loss', 'NFS Loss',
    'WAVG Fair Rindv', 'AVG Fair Rindv', 'WAVG Rindv', 'AVG Rindv', 'NFS Rindv'
]

valores = [
    0.253794730, 0.209656239, 0.333889246, 0.296403408, 0.483972609,
    0.191555917, 0.201281041, 0.346457183, 0.266793668, 0.480675876,
    0.176923722, 0.201010257, 0.350863844, 0.346341550, 0.483585000
]

cores = [
    # 'lightgreen', 'yellowgreen', 'forestgreen', 'green', 'darkgreen', 
    'palegreen', 'lightgreen', 'limegreen', 'forestgreen', 'darkgreen', 
    'lightblue', 'skyblue', 'blue', 'darkblue', 'navy',
    'pink', 'lightcoral', 'red', 'darkred', 'maroon'
]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)
ax.set_xlabel('Injustiça do Grupo')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação e Agrupamentos (Rounds = 5)')
ax.grid(True, linestyle='--', alpha=0.6)  # Adiciona grid
ax.set_xlim(right=max(valores)*1.1)  # Ajusta um limite para eixo X

# Adicionando valores nas barras para facilitar a leitura
for bar, value in zip(bars, valores):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()  # Ajusta o layout para evitar sobreposição de elementos
plt.show()
