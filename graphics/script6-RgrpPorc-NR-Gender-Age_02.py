import matplotlib.pyplot as plt

# Dados dos Rgrp_NR, Rgrp_Gender e Rgrp_Age
metodos = ['AVG', 'AVG Fair', 'WAVG Rindv', 'WAVG Loss', 'WAVG NR', 'WAVG Fair Rdindv', 'WAVG Fair Loss', 'WAVG Fair NR']
Rgrp_NR = [0.473986582, 0.983782317, 0.474120425, 0.464353527, 0.510461425, 0.986606065, 0.985876616, 0.92975816]
Rgrp_Gender = [0.977119344, 0.979279076, 0.964796704, 0.96505796, 0.984400536, 0.982576604, 0.981423666, 0.989664916]
Rgrp_Age = [0, 0.789159768, 0, 0, 0, 0.777583089, 0.782797415, 0.303395644]

# Cálculo da porcentagem em relação ao primeiro valor (AVG) de cada lista
def calcular_porcentagem(valores):
    return [v * 100 for v in valores]

porcentagens_NR = calcular_porcentagem(Rgrp_NR)
porcentagens_Gender = calcular_porcentagem(Rgrp_Gender)
porcentagens_Age = calcular_porcentagem(Rgrp_Age)

# Cores para cada método
cores = ['#8B4513', '#D2691E', '#4169E1', '#1E90FF', '#ADD8E6', '#006400', '#32CD32', '#7CFC00']

# Configuração dos gráficos de barras verticais
fig, axs = plt.subplots(1, 3, figsize=(16, 4))

for i, (dados, titulo) in enumerate(zip([porcentagens_NR, porcentagens_Gender, porcentagens_Age], ['Rgrp NR', 'Rgrp Gender', 'Rgrp Age'])):
    axs[i].bar(range(len(metodos)), dados, color=cores, alpha=0.7)
    axs[i].set_title(f'{titulo}')
    axs[i].grid(axis='y', linestyle='--', linewidth=0.5)
    axs[i].set_ylim(0, 100)  # Definindo a escala do eixo y igual para todos os subplots
    
    if i == 0:
        axs[i].set_ylabel('Porcentagem de Redução da Injustiça do Grupo (%)')  # Apenas no primeiro subplot

# Adicionando a legenda com cores e nomes dos métodos em uma linha
legenda_metodos = [(plt.Rectangle((0, 0), 1, 1, color=cor), metodo) for cor, metodo in zip(cores, metodos)]
fig.legend([leg[0] for leg in legenda_metodos], [leg[1] for leg in legenda_metodos], loc='upper center', ncol=8)

# Omitir os valores numéricos do eixo x
plt.setp(axs, xticks=[])

plt.tight_layout()
plt.show()
