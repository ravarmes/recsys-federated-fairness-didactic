import matplotlib.pyplot as plt

# Dados
categorias = ['FedW-Loss', 'FedW-Loss(π)', 'FedW-Rindv(π)', 'FedW-Rindv', 'FedAVG(π)', 'FedAVG', 'NFS']
rgrp_nr = [0.136963166, 0.303058505, 0.114153242, 0.303121527, 0.123925645, 0.299465100, 0.516497473]
rgrp_gender = [0.000223726, 0.000448324, 0.000295728, 0.000441739, 0.000163247, 0.000346823, 0.022823971]
rgrp_age = [0.008253145, 0.080810894, 0.007272363, 0.081247466, 0.008918054, 0.085916397, 0.023592421]

cores = [
    'lightgreen', 'green',
    'lightblue', 'blue',
    'pink', 'red',
    'black'
]

# Configurações para os subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Gráficos de barras para cada grupo com cores diferentes na horizontal
for i, (grupo, title) in enumerate(zip([rgrp_nr, rgrp_gender, rgrp_age], ['Rgrp Atividade', 'Rgrp Gênero', 'Rgrp Idade'])):
    ax = axs[i]
    if i == 0:
        bars = ax.barh(categorias, grupo, color=cores)
    else:
        bars = ax.barh(categorias, grupo, color=cores)
        ax.yaxis.set_visible(False)  # Ocultar as categorias do eixo y nos subplots 2 e 3
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar, value in zip(bars, grupo):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()
plt.show()
