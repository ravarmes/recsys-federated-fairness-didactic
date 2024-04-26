import numpy as np
import matplotlib.pyplot as plt

categorias = ['FedW-Loss(π)', 'FedW-Rindv(π)', 'FedW-Loss', 'FedW-Rindv', 'FedAVG(π)', 'FedAVG', 'NFS']

valores_NR = [
    [0.007547580, 0.007157760, 0.286251128, 0.281031668, 0.008666780, 0.281103194, 0.534403086],
    [0.188493907, 0.158883184, 0.205435067, 0.210150689, 0.178244501, 0.203020513, 0.447538376],
    [0.214848012, 0.176418781, 0.417489320, 0.418182224, 0.184865654, 0.414271593, 0.567550957]
]

valores_Gender = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

valores_Age = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

data_boxplot = [list(bucket) for bucket in zip(*valores_NR)]

fig, ax = plt.subplots(figsize=(14, 8))

boxprops = dict(facecolor='skyblue', color='black')
medianprops = dict(color='red')
whiskerprops = dict(color='blue', linestyle='--')
capprops = dict(color='green')

bp = ax.boxplot(data_boxplot, labels=categorias, patch_artist=True, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, showmeans=False)

ax.set_ylabel('Injustiça do Grupo', labelpad=10)
plt.title('Distribuição Quantitativa da Injustiça por Métodos de Agregação', pad=10)
plt.grid(True, linestyle='--', alpha=0.3)

# Ajuste da escala no eixo y com base nos valores mínimos e máximos observados
ax.set_ylim(0.0, 0.6)  # Definir os limites para abranger todos os dados
ax.set_yticks(np.arange(0.0, 0.61, 0.05))  # Definir ticks a cada 0.05 para detalhar a escala

plt.tight_layout()
plt.show()
