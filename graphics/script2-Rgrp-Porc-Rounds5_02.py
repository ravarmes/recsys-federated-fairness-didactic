import matplotlib.pyplot as plt

categorias = [
    'WAVG Fair NR', 'AVG Fair NR', 'WAVG NR', 'AVG NR',
    'WAVG Fair Loss', 'AVG Fair Loss', 'WAVG Loss', 'AVG Loss',
    'WAVG Fair Rindv', 'AVG Fair Rindv', 'WAVG Rindv', 'AVG Rindv'
]

porcentagens_reducao = [
    47.56, 56.68, 31.01, 38.76,
    60.15, 58.13, 27.92, 44.50,
    63.41, 58.43, 27.45, 28.38
]

cores = [
    'palegreen', 'lightgreen', 'limegreen', 'forestgreen',
    'lightblue', 'skyblue', 'blue', 'darkblue',
    'pink', 'lightcoral', 'red', 'darkred'
]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, porcentagens_reducao, color=cores)
ax.set_xlabel('Porcentagem de Redução da Injustiça do Grupo (%)')
ax.set_title('Comparação da Redução da Injustiça do Grupo em relação ao Método Não Federado (%)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(right=max(porcentagens_reducao) * 1.1)

# Adicionando linhas para separar os grupos de 4
for i in range(4, len(categorias), 4):
    ax.axhline(i - 0.5, color='black', linewidth=2)

for bar, value in zip(bars, porcentagens_reducao):
    ax.text(value, bar.get_y() + bar.get_height() / 2, f'{value:.2f}%', va='center', ha='left')

plt.tight_layout()
plt.show()
