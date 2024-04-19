import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = ['Inicial', 'Média Aritmética Rindv', 'Média Ponderada Rindv', 'Média Aritmética Fairness Rindv', 'Não Federado Rindv', 
              'Média Aritmética Loss', 'Média Ponderada Loss', 'Média Aritmética Fairness Loss', 'Não Federado Loss',
              'Média Aritmética NR', 'Média Ponderada NR', 'Média Aritmética Fairness NR', 'Não Federado NR']
valores = [0.358072519, 0.583328843, 0.579653025, 0.281483173, 0.560349882, 
           0.001085988, 0.000017779, 0.004058505, 0.003157894, 0.090358347, 
           0.153299317, 0.096040070, 0.254105717]

# Definindo esquema de cores
cores = ['gray',  # Inicial
         'red', 'darkred', 'pink', 'lightcoral',  # Rindv
         'blue', 'darkblue', 'lightblue', 'skyblue',  # Loss
         'green', 'darkgreen', 'lightgreen', 'yellowgreen']  # NR

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)

# Uso de uma escala logarítmica no eixo X
ax.set_xscale('log')

# Definindo rótulos e título do gráfico
ax.set_xlabel('Injustiça do Grupo (Escala Logarítmica)')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação')

plt.tight_layout()
plt.show()
