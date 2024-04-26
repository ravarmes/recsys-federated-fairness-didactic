import matplotlib.pyplot as plt

valores_Rgrp_NR = [0.534403086, 0.281103194, 0.008666780, 0.281031668, 0.286251128, 0.261610925, 0.007157760, 0.007547580, 0.037537456]
valores_Rgrp_Gender = [0.021396633, 0.000489569, 0.000443358, 0.000753232, 0.000747642, 0.000333776, 0.000372802, 0.000397471, 0.000221136]
valores_Rgrp_Age = [0.021389727, 0.088683620, 0.004509815, 0.084792688, 0.081839941, 0.093665794, 0.004757437, 0.004645904, 0.014900177]
valores_RMSE = [1.790414333, 2.308073044, 1.246943116, 2.282620192, 2.249421597, 2.346614122, 1.249799132, 1.253484845, 1.645902157]

categorias_labels = ['NFS', 'AVG', 'WAVG Rindv', 'WAVG Loss', 'WAVG NR', 'AVG Fair', 'WAVG Fair Rindv', 'WAVG Fair Loss', 'WAVG Fair NR']

# Criando o gráfico
fig, ax1 = plt.subplots(figsize=(14, 8))

# Gráfico de Injustiça do Grupo (Rgrp)
ax1.plot(categorias_labels, valores_Rgrp_NR, marker='o', linestyle='-', color='dodgerblue', label='Injustiça do Grupo (Rgrp)', linewidth=2)
# Adicionando valores gerados de Rgrp Gender
ax1.plot(categorias_labels, valores_Rgrp_Gender, marker='s', linestyle='--', color='limegreen', label='Rgrp Gênero', linewidth=2)
# Adicionando valores gerados de Rgrp Age
ax1.plot(categorias_labels, valores_Rgrp_Age, marker='^', linestyle='-.', color='orange', label='Rgrp Idade', linewidth=2)

# Configurando a escala do eixo y para os valores de Rgrp
ax1.set_ylim(0, 0.6)  # Defina os limites do eixo y conforme necessário para destacar os valores de Rgrp

# Configurações do eixo y para Rgrp
ax1.set_ylabel('Injustiça do Grupo (Rgrp)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Comparação de Injustiça do Grupo (Rgrp) e Erro Quadrático Médio (RMSE)')

# Gráfico de Erro Quadrático Médio (RMSE)
ax2 = ax1.twinx()
ax2.plot(categorias_labels, valores_RMSE, marker='s', linestyle='--', color='crimson', label='Erro Quadrático Médio (RMSE)', linewidth=2)
ax2.set_ylabel('Erro Quadrático Médio (RMSE)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Anotação para maior Rgrp
ax1.annotate(f'Menor Rgrp: {min(valores_Rgrp_NR):.2f}', xy=(valores_Rgrp_NR.index(min(valores_Rgrp_NR)), min(valores_Rgrp_NR)), xytext=(10, -10),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.7', color='gray'))

# Anotação para menor RMSE
min_rmse = min(valores_RMSE)
min_rmse_index = valores_RMSE.index(min_rmse)
ax2.annotate(f'Menor RMSE: {min_rmse:.2f}', xy=(min_rmse_index, min_rmse), xytext=(-55, 25),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.9', color='gray'))

# Destacando 'WAVG Fair Loss'
highlight_index = categorias_labels.index('WAVG Fair Loss')
ax1.plot(categorias_labels[highlight_index], valores_Rgrp_NR[highlight_index], 'o', markersize=12, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
ax2.plot(categorias_labels[highlight_index], valores_RMSE[highlight_index], 's', markersize=12, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)

# Legenda combinada para os dois eixos y
lns = ax1.get_lines() + ax2.get_lines()
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right')

plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo X para melhor visualização
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
plt.tight_layout()
plt.show()
