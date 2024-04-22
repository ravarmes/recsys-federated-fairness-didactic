import matplotlib.pyplot as plt
import numpy as np

# Dados, agora listados de baixo para cima para coincidir com a ordem de cima para baixo no gráfico
categorias = [
    'Média Ponderada Fairness NR', 'Média Aritmética Fairness NR', 'Média Ponderada NR', 'Média Aritmética NR', 'Não Federado NR',
    'Média Ponderada Fairness Loss', 'Média Aritmética Fairness Loss', 'Média Ponderada Loss', 'Média Aritmética Loss', 'Não Federado Loss',
    'Média Ponderada Fairness Rindv', 'Média Aritmética Fairness Rindv', 'Média Ponderada Rindv', 'Média Aritmética Rindv', 'Não Federado Rindv'
]

valores = [
    0.003770662, 0.003667223, 0.003752720, 0.003697577, 0.003540524,
    0.003498854, 0.003863712, 0.003662662, 0.003819881, 0.003413113,
    0.003695758, 0.003760421, 0.003736970, 0.003509771, 0.003583554
]

cores = [
    # 'lightgreen', 'yellowgreen', 'forestgreen', 'green', 'darkgreen', 
    'palegreen', 'lightgreen', 'limegreen', 'forestgreen', 'darkgreen', 
    'lightblue', 'skyblue', 'blue', 'darkblue', 'navy',
    'pink', 'lightcoral', 'red', 'darkred', 'maroon'
]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)
ax.set_xlabel('Injustiça do Grupo (Rounds = 5)')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação')
ax.grid(True, linestyle='--', alpha=0.6)  # Adiciona grid
ax.set_xlim(right=max(valores)*1.1)  # Ajusta um limite para eixo X

# Adicionando valores nas barras para facilitar a leitura
for bar, value in zip(bars, valores):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()  # Ajusta o layout para evitar sobreposição de elementos
plt.show()
