import matplotlib.pyplot as plt
import numpy as np

# Dados de Injustiça do Grupo (Rgrp) e Erro Quadrático Médio (RMSE)
valores_Rgrp = [
    0.003770662, 0.003667223, 0.003752720, 0.003697577, 0.003540524,
    0.003498854, 0.003863712, 0.003662662, 0.003819881, 0.003413113,
    0.003695758, 0.003760421, 0.003736970, 0.003509771, 0.003583554
]

valores_RMSE = [
    1.230573416, 1.230862379, 1.231484532, 1.231635094, 1.232785702,
    1.231122136, 1.230731368, 1.231925964, 1.230286479, 1.232954264,
    1.230955720, 1.230825067, 1.231920481, 1.230504751, 1.232975006
]

categorias = [
    'Média Ponderada Fairness NR', 'Média Aritmética Fairness NR', 'Média Ponderada NR', 'Média Aritmética NR', 'Não Federado NR',
    'Média Ponderada Fairness Loss', 'Média Aritmética Fairness Loss', 'Média Ponderada Loss', 'Média Aritmética Loss', 'Não Federado Loss',
    'Média Ponderada Fairness Rindv', 'Média Aritmética Fairness Rindv', 'Média Ponderada Rindv', 'Média Aritmética Rindv', 'Não Federado Rindv'
]

# Separando as categorias por tipo
categorias_NR = categorias[0:5]
categorias_Loss = categorias[5:10]
categorias_Rindv = categorias[10:15]

# Criando subplots para cada conjunto de categorias
fig, axs = plt.subplots(3, 1, figsize=(14, 16))

# Subplot para NR
axs[0].plot(categorias_NR, valores_Rgrp[0:5], marker='o', color='steelblue', label='Injustiça do Grupo (Rgrp)')
axs[0].plot(categorias_NR, valores_RMSE[0:5], marker='s', color='lightcoral', label='Erro Quadrático Médio (RMSE)')
axs[0].set_title('Categorias NR')
axs[0].set_ylabel('Valores')
axs[0].legend()
axs[0].tick_params(axis='x', rotation=45)  # Rotaciona rótulos do eixo x

# Subplot para Loss
axs[1].plot(categorias_Loss, valores_Rgrp[5:10], marker='o', color='steelblue', label='Injustiça do Grupo (Rgrp)')
axs[1].plot(categorias_Loss, valores_RMSE[5:10], marker='s', color='lightcoral', label='Erro Quadrático Médio (RMSE)')
axs[1].set_title('Categorias Loss')
axs[1].set_ylabel('Valores')
axs[1].legend()
axs[1].tick_params(axis='x', rotation=45)  # Rotaciona rótulos do eixo x

# Subplot para Rindv
axs[2].plot(categorias_Rindv, valores_Rgrp[10:15], marker='o', color='steelblue', label='Injustiça do Grupo (Rgrp)')
axs[2].plot(categorias_Rindv, valores_RMSE[10:15], marker='s', color='lightcoral', label='Erro Quadrático Médio (RMSE)')
axs[2].set_title('Categorias Rindv')
axs[2].set_ylabel('Valores')
axs[2].legend()
axs[2].tick_params(axis='x', rotation=45)  # Rotaciona rótulos do eixo x

plt.tight_layout()
plt.show()
