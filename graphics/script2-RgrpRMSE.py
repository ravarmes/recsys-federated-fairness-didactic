import matplotlib.pyplot as plt
import numpy as np

valores_Rgrp = [0.529993197, 0.298492094, 0.320935562, 0.316976041, 0.306027815, 0.222787817, 0.176208914, 0.197502539, 0.272373081]
valores_RMSE = [1.767392119, 2.115062714, 2.307729245, 2.229744196, 2.2338444, 1.614019354, 1.637112022, 1.545639038, 1.528500498]
categorias_labels = ['NFS', 'AVG', 'WAVG Rindv', 'WAVG Loss', 'WAVG NR', 'AVG Fair', 'WAVG Fair Rindv', 'WAVG Fair Loss', 'WAVG Fair NR']

# Criando o gráfico
fig, ax1 = plt.subplots(figsize=(14, 8))

# Gráfico de Injustiça do Grupo (Rgrp)
ax1.plot(categorias_labels, valores_Rgrp, marker='o', linestyle='-', color='dodgerblue', label='Injustiça do Grupo (Rgrp)', linewidth=2)
# ax1.set_xlabel('Categorias')
ax1.set_ylabel('Injustiça do Grupo (Rgrp)', color='dodgerblue')
ax1.tick_params(axis='y', labelcolor='dodgerblue')
ax1.set_title('Comparação de Injustiça do Grupo (Rgrp) e Erro Quadrático Médio (RMSE)')

# Gráfico de Erro Quadrático Médio (RMSE)
ax2 = ax1.twinx()
ax2.plot(categorias_labels, valores_RMSE, marker='s', linestyle='--', color='crimson', label='Erro Quadrático Médio (RMSE)', linewidth=2)
ax2.set_ylabel('Erro Quadrático Médio (RMSE)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

# Anotação para maior Rgrp
ax1.annotate(f'Menor Rgrp: {min(valores_Rgrp):.2f}', xy=(valores_Rgrp.index(min(valores_Rgrp)), min(valores_Rgrp)), xytext=(10, -10),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.7', color='gray'))

# Anotação para menor RMSE
min_rmse = min(valores_RMSE)
min_rmse_index = valores_RMSE.index(min_rmse)
ax2.annotate(f'Menor RMSE: {min_rmse:.2f}', xy=(min_rmse_index, min_rmse), xytext=(-55, 25),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.9', color='gray'))
             #textcoords='offset points', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=2), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5', color='black'))

# # Anotação para 'WAVG Fair Loss'
# # Destacando 'WAVG Fair Loss'
# highlight_index = categorias_labels.index('WAVG Fair Loss')
# ax1.plot(categorias_labels[highlight_index], valores_Rgrp[highlight_index], marker='o', markersize=10, markeredgecolor='gold', markerfacecolor='green')
# ax2.plot(categorias_labels[highlight_index], valores_RMSE[highlight_index], marker='s', markersize=10, markeredgecolor='gold', markerfacecolor='green')
# ax1.annotate('Destaque', xy=(highlight_index, valores_Rgrp[highlight_index]), xytext=(highlight_index, valores_Rgrp[highlight_index]+0.05),
#              textcoords='offset points', arrowprops=dict(arrowstyle='->', color='green'))

# Destacando 'WAVG Fair Loss'
highlight_index = categorias_labels.index('WAVG Fair Loss')
ax1.plot(categorias_labels[highlight_index], valores_Rgrp[highlight_index], 'o', markersize=12, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
ax2.plot(categorias_labels[highlight_index], valores_RMSE[highlight_index], 's', markersize=12, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)

# Legenda combinada para os dois eixos y
lns = ax1.get_lines() + ax2.get_lines()
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right')

plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo X para melhor visualização
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
plt.tight_layout()
plt.show()
