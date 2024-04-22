import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = [
    'WAVG Fair', 'AVG Fair', 'WAVG', 'AVG'
]

porcentagens_reducao = [
    47.56, 56.68, 31.01, 38.76,
    60.15, 58.13, 27.92, 44.50,
    63.41, 58.43, 27.45, 28.38
]

# Reorganizando os dados em uma matriz 3x4 para o heatmap
dados_heatmap = np.array(porcentagens_reducao).reshape(3, 4)

# Configurações do heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(dados_heatmap, cmap='coolwarm')

ax.set_xticks(np.arange(len(categorias)))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(categorias, rotation=0, ha='center')
ax.set_yticklabels(['NR', 'Loss', 'Rindv'])

for i in range(3):
    for j in range(4):
        text = ax.text(j, i, f'{dados_heatmap[i, j]:.2f}%', ha='center', va='center', color='black')

plt.colorbar(im, ax=ax)
plt.title('Heatmap de Redução da Injustiça do Grupo em relação ao Método Não Federado (%)')
plt.tight_layout()
plt.show()
