import matplotlib.pyplot as plt
import numpy as np

# Dados de Injustiça do Grupo
valores_Rgrp = [
    0.003770662, 0.003667223, 0.003752720, 0.003697577, 0.003540524,
    0.003498854, 0.003863712, 0.003662662, 0.003819881, 0.003413113,
    0.003695758, 0.003760421, 0.003736970, 0.003509771, 0.003583554
]

# Dados de Erro Quadrático Médio (RMSE)
valores_RMSE = [
    1.230573416, 1.230862379, 1.231484532, 1.231635094, 1.232785702,
    1.231122136, 1.230731368, 1.231925964, 1.230286479, 1.232954264,
    1.230955720, 1.230825067, 1.231920481, 1.230504751, 1.232975006
]

categorias = np.arange(len(valores_Rgrp))  # Categorias para o eixo x

fig, ax1 = plt.subplots(figsize=(14, 8))

# Gráfico de Injustiça do Grupo (Rgrp)
ax1.plot(categorias, valores_Rgrp, marker='o', color='blue', label='Injustiça do Grupo (Rgrp)')
ax1.set_xlabel('Índice')
ax1.set_ylabel('Injustiça do Grupo (Rgrp)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Comparação de Injustiça do Grupo (Rgrp) e Erro Quadrático Médio (RMSE)')

# Gráfico de Erro Quadrático Médio (RMSE)
ax2 = ax1.twinx()  # Segundo eixo y compartilhando o mesmo eixo x
ax2.plot(categorias, valores_RMSE, marker='s', color='red', label='Erro Quadrático Médio (RMSE)')
ax2.set_ylabel('Erro Quadrático Médio (RMSE)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Legenda combinada para os dois eixos y
lns = ax1.get_lines() + ax2.get_lines()
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right')

plt.xticks(categorias, categorias)  # Adiciona os índices ao eixo x
plt.grid(True)
plt.tight_layout()
plt.show()
