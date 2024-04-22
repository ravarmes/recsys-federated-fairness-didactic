import matplotlib.pyplot as plt
import numpy as np

# Dados, agora listados de baixo para cima para coincidir com a ordem de cima para baixo no gráfico
categorias = [
    'Média Ponderada Fairness NR', 'Média Aritmética Fairness NR', 'Média Ponderada NR', 'Média Aritmética NR', 'Não Federado NR',
    'Média Ponderada Fairness Loss', 'Média Aritmética Fairness Loss', 'Média Ponderada Loss', 'Média Aritmética Loss', 'Não Federado Loss',
    'Média Ponderada Fairness Rindv', 'Média Aritmética Fairness Rindv', 'Média Ponderada Rindv', 'Média Aritmética Rindv', 'Não Federado Rindv'
]

valores = [
    1.230573416, 1.230862379, 1.231484532, 1.231635094, 1.232785702,
    1.231122136, 1.230731368, 1.231925964, 1.230286479, 1.232954264,
    1.230955720, 1.230825067, 1.231920481, 1.230504751, 1.232975006
]

cores = [
    # 'lightgreen', 'yellowgreen', 'forestgreen', 'green', 'darkgreen', 
    'palegreen', 'lightgreen', 'limegreen', 'forestgreen', 'darkgreen', 
    'lightblue', 'skyblue', 'blue', 'darkblue', 'navy',
    'pink', 'lightcoral', 'red', 'darkred', 'maroon'
]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)
ax.set_xlabel('Erro quadrático médio (Rounds = 5)')
ax.set_title('Comparação do Erro Quadrático Médio entre Diferentes Métodos de Agregação')
ax.grid(True, linestyle='--', alpha=0.2, color='gray')  # Ajustando a cor e transparência das linhas de grade

# Ajuste automático dos limites do eixo x com uma pequena margem
ax.set_xlim(right=max(valores) * 1.05)  # Aumentando o limite do eixo X em 5% além do maior valor

# Adicionando valores nas barras
for bar, value in zip(bars, valores):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()  # Ajustando o layout
plt.show()



