import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = [
    'Média Aritmética Fairness', 'Média Ponderada', 'Média Aritmética', 'Não Federado'
]
subcategorias = ['Rindv', 'Loss', 'NR']
valores_por_grupo = np.array([
    [0.096040070, 0.153299317, 0.090358347, 0.254105717],
    [0.004058505, 0.000017779, 0.001085988, 0.003157894],
    [0.281483173, 0.579653025, 0.583328843, 0.560349882],
])

cores = ['lightgreen', 'lightblue', 'pink']

largura_barra = 0.2
n_barras = len(subcategorias)
indice = np.arange(len(categorias))

fig, ax = plt.subplots(figsize=(10, 6))

# Criando as barras para cada subcategoria em cada categoria
for i in range(n_barras):
    ax.bar(indice + i * largura_barra, valores_por_grupo[i], width=largura_barra, label=subcategorias[i], color=cores[i])

ax.set_xlabel('Grupos das Barras')
ax.set_ylabel('Valores')
ax.set_title('Comparação de Categorias com Subcategorias')
ax.set_xticks(indice + largura_barra * (n_barras - 1) / 2)
ax.set_xticklabels(categorias)
ax.legend()

plt.tight_layout()
plt.show()
