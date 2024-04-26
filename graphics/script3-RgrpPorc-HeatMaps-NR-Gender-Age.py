import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = ['AVG', 'AVG Fair', 'WAVG', 'WAVG Fair']

# Reduções
porcentagens_reducao_NR = [
    42.02, 76.01, 41.31, 77.90,
    42.02, 76.01, 41.32, 73.48
]

porcentagens_reducao_Gender = [
    98.48, 99.28, 98.06, 98.70,
    98.48, 99.28, 98.04, 99.02
]

porcentagens_reducao_Age = [
    0, 62.20, 0, 69.18,
    0, 62.20, 0, 65.02
]

# DataSet dos Heatmaps
datasets = [
    np.array(porcentagens_reducao_NR).reshape(2, 4),
    np.array(porcentagens_reducao_Gender).reshape(2, 4),
    np.array(porcentagens_reducao_Age).reshape(2, 4)
]

titles = ['Redução de Rgrp :: Agrupamento por Atividade', 'Redução de Rgrp :: Agrupamento por Gênero', 'Redução de Rgrp :: Agrupamento por Idade']

# Configurações do gráfico
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Loop para criar cada subplot de heatmap
for ax, data, title in zip(axs, datasets, titles):
    im = ax.imshow(data, cmap='coolwarm')

    ax.set_xticks(np.arange(len(categorias)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(categorias)
    ax.set_yticklabels(['Loss', 'Rindv'])
    ax.set_title(title)

    # Adicionando texto dentro dos quadrados
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]:.2f}%', ha='center', va='center', color='black')

# Ajuste geral do layout
fig.tight_layout()

# Adicionando uma colorbar que é comum para todos os heatmaps
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # left, bottom, width, height (in figure coordinate)
fig.colorbar(im, cax=cbar_ax)

plt.show()
