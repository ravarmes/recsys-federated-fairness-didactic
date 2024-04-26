import matplotlib.pyplot as plt

# Dados
categorias = ['WAVG Fair', 'WAVG', 'AVG Fair', 'AVG', 'NFS']

valores = [
    [(0.176923722+0.191555917+0.253794730)/3, (0.350863844+0.346457183+0.333889246)/3, 0.203982512, 0.303179542, 0.482744495],
    [(0.175494105+0.203449160+0.290951431)/3, (0.291007280+0.287494898+0.278166384)/3, 0.241593122, 0.293804646, 0.577241898]
]

# Preparando os dados para o boxplot
data_boxplot = [list(bucket) for bucket in zip(*valores)]

# Configurações do boxplot
# fig, ax = plt.subplots(figsize=(10, 9))
fig, ax = plt.subplots(figsize=(14, 8))

boxprops = dict(facecolor='skyblue', color='black')
medianprops = dict(color='red')
whiskerprops = dict(color='blue', linestyle='--')
capprops = dict(color='green')

bp = ax.boxplot(data_boxplot, labels=categorias, patch_artist=True, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, showmeans=False)

# ax.set_xlabel('Métodos de Agregação', labelpad=10)
ax.set_ylabel('Injustiça do Grupo', labelpad=10)

plt.title('Distribuição Quantitativa da Injustiça por Métodos de Agregação', pad=10)
plt.grid(True, linestyle='--', alpha=0.3)

# Ajuste da escala no eixo y com intervalos de 0.025
ax.set_yticks([i / 100 for i in range(19, 60, 2)])  # Intervalo de 0.025 de 0.175 a 0.49

plt.tight_layout()
plt.show()
