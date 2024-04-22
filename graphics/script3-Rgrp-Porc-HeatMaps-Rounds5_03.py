import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = [
    'WAVG Fair', 'AVG Fair/AVG', 'WAVG', 'AVG'
]

porcentagens_reducao = [
    47.56, 58.13, 31.01, 38.76,
    60.15, 27.92, 27.45, 38.76
]

# Reorganizando os dados em uma matriz 2x4 para o heatmap
dados_heatmap = np.array(porcentagens_reducao).reshape(2, 4)

# Configurações do heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(dados_heatmap, cmap='coolwarm')

ax.set_xticks(np.arange(len(categorias)))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(categorias, rotation=0, ha='center')
ax.set_yticklabels(['NR', 'Loss'])

for i in range(2):
    for j in range(4):
        text = ax.text(j, i, f'{dados_heatmap[i, j]:.2f}%', ha='center', va='center', color='black')

plt.colorbar(im, ax=ax)
plt.title('Heatmap de Redução da Injustiça do Grupo em relação ao Método Não Federado (%)')
plt.tight_layout()
plt.show()
