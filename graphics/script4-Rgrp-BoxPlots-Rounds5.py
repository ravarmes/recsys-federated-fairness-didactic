import matplotlib.pyplot as plt

# Dados
categorias = ['WAVG Fair', 'AVG Fair', 'WAVG', 'AVG', 'NFS']

valores = [
    [0.253794730, 0.209656239, 0.333889246, 0.296403408, 0.483972609],
    [0.191555917, 0.201281041, 0.346457183, 0.266793668, 0.480675876],
    [0.176923722, 0.201010257, 0.350863844, 0.346341550, 0.483585000]
]

# Preparando os dados para o boxplot
data_boxplot = [list(bucket) for bucket in zip(*valores)]

# Configurações do boxplot
fig, ax = plt.subplots(figsize=(10, 9))

boxprops = dict(facecolor='skyblue', color='black')
medianprops = dict(color='red')
whiskerprops = dict(color='blue', linestyle='--')
capprops = dict(color='green')

bp = ax.boxplot(data_boxplot, labels=categorias, patch_artist=True, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, showmeans=False)

# ax.set_xlabel('Métodos de Agregação', labelpad=10)
ax.set_ylabel('Injustiça do Grupo', labelpad=10)

plt.title('Boxplot da Injustiça do Grupo por Métodos de Agregação', pad=10)
plt.grid(True, linestyle='--', alpha=0.3)

# Ajuste da escala no eixo y com intervalos de 0.025
ax.set_yticks([i / 100 for i in range(17, 49, 2)])  # Intervalo de 0.025 de 0.175 a 0.49

plt.tight_layout()
plt.show()
