import numpy as np
import matplotlib.pyplot as plt

categorias = ['FedW-Loss(π)', 'FedW-Rindv(π)', 'FedW-Loss', 'FedW-Rindv', 'FedAVG(π)', 'FedAVG', 'NFS']

valores_NR = [
    [0.007547580, 0.007157760, 0.286251128, 0.281031668, 0.008666780, 0.281103194, 0.534403086],
    [0.188493907, 0.158883184, 0.205435067, 0.210150689, 0.178244501, 0.203020513, 0.447538376],
    [0.214848012, 0.176418781, 0.417489320, 0.418182224, 0.184865654, 0.414271593, 0.567550957]
]

valores_Gender = [
    [0.000397471, 0.000372802, 0.000747642, 0.000753232, 0.000443358, 0.000489569, 0.021396633],
    [0.000012787, 0.000013031, 0.000520241, 0.000491322, 0.000007853, 0.000513757, 0.021150737],
    [0.000260919, 0.000501350, 0.000077088, 0.000080662, 0.000038531, 0.000037142, 0.021924542]
]

valores_Age = [
    [0.000397471, 0.000372802, 0.000747642, 0.000753232, 0.000443358, 0.000489569, 0.021396633],
    [0.000012787, 0.000013031, 0.000520241, 0.000491322, 0.000007853, 0.000513757, 0.021150737],
    [0.000260919, 0.000501350, 0.000077088, 0.000080662, 0.000038531, 0.000037142, 0.021924542]
]


data_boxplot_NR = [list(bucket) for bucket in zip(*valores_NR)]
data_boxplot_Gender = [list(bucket) for bucket in zip(*valores_Gender)]
data_boxplot_Age = [list(bucket) for bucket in zip(*valores_Age)]

fig, ax = plt.subplots(figsize=(14, 8))

# Definição de cores e propriedades
boxprops_NR = dict(facecolor='skyblue', color='black')
boxprops_Gender = dict(facecolor='lightgreen', color='black')
boxprops_Age = dict(facecolor='salmon', color='black')
medianprops = dict(color='red')
whiskerprops = dict(color='blue', linestyle='--')
capprops = dict(color='green')

# Adicionar os boxplots para NR
bp_NR = ax.boxplot(data_boxplot_NR, positions=np.array(range(len(categorias)))*3, widths=0.6, patch_artist=True, boxprops=boxprops_NR, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, showmeans=False)

# Criar um segundo eixo y para Gender e Age
ax2 = ax.twinx()
bp_Gender = ax2.boxplot(data_boxplot_Gender, positions=np.array(range(len(categorias)))*3 + 1, widths=0.6, patch_artist=True, boxprops=boxprops_Gender, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, showmeans=False)
bp_Age = ax2.boxplot(data_boxplot_Age, positions=np.array(range(len(categorias)))*3 + 2, widths=0.6, patch_artist=True, boxprops=boxprops_Age, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, showmeans=False)

# Configurações de eixo y
ax.set_ylabel('Injustiça do Grupo (Atividade)', labelpad=10)
ax2.set_ylabel('Injustiça do Grupo (Gênero e Idade)', labelpad=10)

# Exibição da legenda
ax.legend([bp_NR["boxes"][0], bp_Gender["boxes"][0], bp_Age["boxes"][0]], ['Atividade', 'Gênero', 'Idade'], loc='upper left')

# Restante das configurações
plt.title('Distribuição Quantitativa da Injustiça por Métodos de Agregação', pad=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_ylim(0.0, 0.6)
ax.set_yticks(np.arange(0.0, 0.61, 0.05))
ax2.set_ylim(0, 0.022)  # Ajuste para o segundo eixo y

# Definindo as categorias como rótulos para o eixo x
categorias = ['FedW-Loss(π)', 'FedW-Rindv(π)', 'FedW-Loss', 'FedW-Rindv', 'FedAVG(π)', 'FedAVG', 'NFS']

# Atualizando o rótulo do eixo x
ax.set_xticks(np.array(range(len(categorias)))*3 + 1)
ax.set_xticklabels(categorias)

plt.tight_layout()
plt.show()
