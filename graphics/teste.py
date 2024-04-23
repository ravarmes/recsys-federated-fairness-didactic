import matplotlib.pyplot as plt
import numpy as np

valores_Rgrp = [0.529993197, 0.298492094, 0.320935562, 0.316976041, 0.306027815, 0.222787817, 0.176208914, 0.197502539, 0.272373081]
valores_RMSE = [1.767392119, 2.115062714, 2.307729245, 2.229744196, 2.2338444, 1.614019354, 1.637112022, 1.545639038, 1.528500498]
categorias_labels = ['NFS', 'AVG', 'WAVG Rindv', 'WAVG Loss', 'WAVG NR', 'AVG Fair', 'WAVG Fair Rindv', 'WAVG Fair Loss', 'WAVG Fair NR']

# Criando o gráfico
fig, ax1 = plt.subplots(figsize=(14, 8))

# Gráfico de Injustiça do Grupo (Rgrp)
ax1.plot(categorias_labels, valores_Rgrp, marker='o', linestyle='-', color='dodgerblue', label='Injustiça do Grupo (Rgrp)', linewidth=2)
ax2 = ax1.twinx()
ax2.plot(categorias_labels, valores_RMSE, marker='s', linestyle='--', color='crimson', label='Erro Quadrático Médio (RMSE)', linewidth=2)

# Destacando 'WAVG Fair Loss'
highlight_index = categorias_labels.index('WAVG Fair Loss')
ax1.plot(categorias_labels[highlight_index], valores_Rgrp[highlight_index], 'o', markersize=12, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
ax2.plot(categorias_labels[highlight_index], valores_RMSE[highlight_index], 's', markersize=12, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)

# # Anotação auxiliar com círculos
# circle_rgrp = plt.Circle((highlight_index, valores_Rgrp[highlight_index]), 0.08, color='green', fill=False, clip_on=True, linewidth=2)
# circle_rmse = plt.Circle((highlight_index, valores_RMSE[highlight_index]), 0.08, color='green', fill=False, clip_on=True, linewidth=2)
# ax1.add_artist(circle_rgrp)
# ax2.add_artist(circle_rmse)

ax1.set_ylabel('Injustiça do Grupo (Rgrp)', color='dodgerblue')
ax1.tick_params(axis='y', labelcolor='dodgerblue')
ax2.set_ylabel('Erro Quadrático Médio (RMSE)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')
ax1.set_title('Comparação de Injustiça do Grupo e Erro Quadrático Médio')

# Legenda e outros ajustes
lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.xticks(rotation=45)
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
plt.tight_layout()
plt.show()
