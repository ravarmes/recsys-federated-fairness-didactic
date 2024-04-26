import matplotlib.pyplot as plt
import numpy as np

# Dados
categorias = ['AVG', 'AVG Fair', 'WAVG', 'WAVG Fair']

porcentagens_reducao = [
    42.02, 76.01, 41.31, 77.90,
    42.02, 76.01, 41.32, 73.48
]

# Reorganizando os dados em uma matriz 3x4 para o heatmap
dados_heatmap = np.array(porcentagens_reducao).reshape(2, 4)

# Configurações do heatmap
#fig, ax = plt.subplots(figsize=(10, 6))
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(dados_heatmap, cmap='coolwarm')

ax.set_xticks(np.arange(len(categorias)))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(categorias, rotation=0, ha='center')
ax.set_yticklabels(['Loss', 'Rindv'])

# Adicionando texto somente onde é necessário
for i in range(2):
    for j in range(4):
        # Aqui, omitimos as posições que não queremos mostrar (posição [1,1] para 58.13, [1,3] para 38.76)
        if (i, j) not in [(0, 0), (0, 1)]:
            if (i, j) in [(1, 0), (1, 1)]:
                ax.text(j, i-0.5, f'{dados_heatmap[i, j]:.2f}%', ha='center', va='center', color='black')
            else:
                ax.text(j, i, f'{dados_heatmap[i, j]:.2f}%', ha='center', va='center', color='black')
        
plt.colorbar(im, ax=ax)
plt.title('Heatmap de Redução da Injustiça do Grupo em relação ao Método Não Federado (%)')
plt.tight_layout()
plt.show()
