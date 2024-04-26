import matplotlib.pyplot as plt

# Dados
valores_Rgrp_NR = [0.516497473, 0.299465100, 0.303121527, 0.303058505, 0.123925645, 0.114153242, 0.136963166]
valores_Rgrp_Gender = [0.022823971, 0.000346823, 0.000441739, 0.000448324, 0.000163247, 0.000295728, 0.000223726]
valores_Rgrp_Age = [0.023592421, 0.085916397, 0.081247466, 0.080810894, 0.008918054, 0.007272363, 0.008253145]
valores_RMSE = [1.778912981, 2.278090557, 2.242979129, 2.233993689, 1.491185466, 1.441035231, 1.462143381]
categorias_labels = ['NFS', 'FedAVG', 'FedW-Rindv', 'FedW-Loss', 'FedAVG(π)', 'FedW-Rindv(π)', 'FedW-Loss(π)']

fig, axs = plt.subplots(3, 1, figsize=(14, 10))

legend_labels = ['Injustiça do Grupo (Rgrp)', 'Erro Quadrático Médio (RMSE)']
legend_labels_Rgrp = ['Rgrp Atividade', 'Rgrp Gênero', 'Rgrp Idade']

for i, (grupo, rmse) in enumerate(zip([valores_Rgrp_NR, valores_Rgrp_Gender, valores_Rgrp_Age], [valores_RMSE]*3)):
    ax = axs[i]
    if i < 2:
        ax.plot(range(len(grupo)), grupo, marker='o', linestyle='-', color='dodgerblue', label=legend_labels_Rgrp[i], linewidth=2)
    else:
        ax.plot(categorias_labels, grupo, marker='o', linestyle='-', color='dodgerblue', label=legend_labels_Rgrp[i], linewidth=2)
        ax.set_xticks(range(len(categorias_labels)))
        ax.set_xticklabels(categorias_labels)
        
    if i == 1:
        ax.set_ylabel('Injustiça do Grupo (Rgrp)', color='dodgerblue')
    ax.tick_params(axis='y', labelcolor='dodgerblue')
    ax2 = ax.twinx()
    if i == 0 or i == 2:
        ax2.plot(range(len(rmse)), rmse, marker='s', linestyle='--', color='crimson', label=legend_labels[1], linewidth=2)
        ax2.tick_params(axis='y', labelcolor='crimson')
    else:
        ax2.plot(categorias_labels, rmse, marker='s', linestyle='--', color='crimson', label=legend_labels[1], linewidth=2)
        ax2.set_ylabel('Erro Quadrático Médio (RMSE)', color='crimson')
        ax2.tick_params(axis='y', labelcolor='crimson')
        
    if i < 2:
        ax2.set_xticks([])
    if i == 0:
        ax.set_title('Comparação de Injustiça do Grupo (Rgrp) e Erro Quadrático Médio (RMSE)')

    menor_valor = min(grupo)
    index_menor_valor = grupo.index(menor_valor)

    axs[i].annotate(f'Menor Rgrp: {menor_valor:.5f}', xy=(index_menor_valor, menor_valor), xytext=(10, 30),
                    textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='gray'))

    # Adicionando o texto indicativo no subplot
    # ax.text(0.5, 1.05, subplot_titles[i], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
