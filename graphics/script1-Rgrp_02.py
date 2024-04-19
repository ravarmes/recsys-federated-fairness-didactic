import matplotlib.pyplot as plt
import numpy as np

# Dados, agora listados de baixo para cima para coincidir com a ordem de cima para baixo no gráfico
categorias = [
    'Média Aritmética Fairness NR', 'Média Ponderada NR', 'Média Aritmética NR', 'Não Federado NR',
    'Média Aritmética Fairness Loss', 'Média Ponderada Loss', 'Média Aritmética Loss', 'Não Federado Loss',
    'Média Aritmética Fairness Rindv', 'Média Ponderada Rindv', 'Média Aritmética Rindv', 'Não Federado Rindv',
    'Inicial'
]

valores = [
    0.096040070, 0.153299317, 0.090358347, 0.254105717,
    0.004058505, 0.000017779, 0.001085988, 0.003157894,
    0.281483173, 0.579653025, 0.583328843, 0.560349882,
    0.358072519
]

cores = [
    'lightgreen', 'darkgreen', 'green', 'yellowgreen',
    'lightblue', 'darkblue', 'blue', 'skyblue',
    'pink', 'darkred', 'red', 'lightcoral',
    'gray'
]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)
ax.set_xlabel('Injustiça do Grupo')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação')
ax.grid(True, linestyle='--', alpha=0.6)  # Adiciona grid
ax.set_xlim(right=max(valores)*1.1)  # Ajusta um limite para eixo X

# Adicionando valores nas barras para facilitar a leitura
for bar, value in zip(bars, valores):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()  # Ajusta o layout para evitar sobreposição de elementos
plt.show()
