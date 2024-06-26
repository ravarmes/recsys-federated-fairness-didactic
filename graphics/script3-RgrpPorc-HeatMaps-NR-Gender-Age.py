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

datasets = [
    np.array(porcentagens_reducao_NR).reshape(2, 4),
    np.array(porcentagens_reducao_Gender).reshape(2, 4),
    np.array(porcentagens_reducao_Age).reshape(2, 4)
]

titles = ['Atividade', 'Gênero', 'Idade']

# Configurações do gráfico
# fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # Figura estreita
fig, axs = plt.subplots(1, 3, figsize=(18, 6))


# Ajuste para deixar espaço para o título geral
plt.subplots_adjust(top=0.9, right=0.85)

# Adicionar título geral para a figura
fig.suptitle("Análise da Redução de Rgrp por Diferentes Agrupamentos", fontsize=12, y=0.70)  # Posição mais baixa do título geral

# Criar os subplots
for ax, data, title in zip(axs, datasets, titles):
    im = ax.imshow(data, cmap='coolwarm')
    ax.set_xticks(np.arange(len(categorias)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(categorias)
    ax.set_yticklabels(['Loss', 'Rindv'])
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Aqui, omitimos as posições que não queremos mostrar (posição [1,1] para 58.13, [1,3] para 38.76)
            if (i, j) not in [(0, 0), (0, 1)]:
                if (i, j) in [(1, 0), (1, 1)]:
                    ax.text(j, i-0.5, f'{data[i, j]:.2f}%', ha='center', va='center', color='black')
                else:
                    ax.text(j, i, f'{data[i, j]:.2f}%', ha='center', va='center', color='black')

# Ajustar a altura da barra lateral (colorbar)
#cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # Alinhado com a altura dos subplots
cbar_ax = fig.add_axes([0.88, 0.395, 0.03, 0.22])  # Reduz a altura da barra de cor

fig.colorbar(im, cax=cbar_ax)  # Adiciona a barra de cor

plt.show()  # Exibe o gráfico
