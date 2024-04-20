import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = ['Média Aritmética', 'Média Ponderada', 'Não Federado']
metodos = ['Fairness NR', 'Loss NR', 'Rindv NR', 'Fairness Loss', 'Loss Loss', 'Rindv Loss']
valores = np.array([
    [0.096040070, 0.153299317, 0.090358347],
    [0.254105717, 0.004058505, 0.000017779],
    [0.001085988, 0.003157894, 0.281483173],
    [0.579653025, 0.583328843, 0.560349882],
])

cores = ['lightgreen', 'lightblue', 'pink', 'lightcoral', 'skyblue', 'red']
largura_barra = 0.15
n_barras = len(metodos)
indice = np.arange(len(categorias))

fig, ax = plt.subplots(figsize=(10, 6))

# Criando as barras para cada categoria
for i in range(min(n_barras, valores.shape[1])):
    ax.barh(indice + (i - n_barras/2) * largura_barra, valores[i], height=largura_barra, label=metodos[i], color=cores[i])

ax.set_xlabel('Injustiça do Grupo')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação')
ax.set_yticks(indice)
ax.set_yticklabels(categorias)
ax.invert_yaxis()  # Inverte a ordem dos métodos
ax.legend()

plt.tight_layout()
plt.show()
