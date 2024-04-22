import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = ['AVG', 'AVG Fair', 'WAVG', 'WAVG Fair']

porcentagens_reducao = [
    38.76, 58.13, 31.01, 47.56,
    38.76, 58.13, 27.92, 60.15,
    38.76, 58.13, 27.45, 63.41
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

# Adicionando texto somente onde é necessário
for i in range(3):
    for j in range(4):
        # Aqui, omitimos as posições que não queremos mostrar (posição [1,1] para 58.13, [1,3] para 38.76)
        if (i, j) not in [(0, 0), (2, 0), (0, 1), (2, 1)]:
            ax.text(j, i, f'{dados_heatmap[i, j]:.2f}%', ha='center', va='center', color='black')

plt.colorbar(im, ax=ax)
plt.title('Heatmap de Redução da Injustiça do Grupo em relação ao Método Não Federado (%)')
plt.tight_layout()
plt.show()
