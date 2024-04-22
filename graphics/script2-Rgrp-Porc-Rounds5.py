import matplotlib.pyplot as plt

# Dados
categorias = [
    'Média Ponderada Fairness NR', 'Média Aritmética Fairness NR', 'Média Ponderada NR', 'Média Aritmética NR',
    'Média Ponderada Fairness Loss', 'Média Aritmética Fairness Loss', 'Média Ponderada Loss', 'Média Aritmética Loss',
    'Média Ponderada Fairness Rindv', 'Média Aritmética Fairness Rindv', 'Média Ponderada Rindv', 'Média Aritmética Rindv'
]

porcentagens_reducao = [
    47.56, 56.68, 31.01, 38.76,
    60.15, 58.13, 27.92, 44.50,
    63.41, 58.43, 27.45, 28.38
]

# Cores padrão
cores = [
    'palegreen', 'lightgreen', 'limegreen', 'forestgreen',
    'lightblue', 'skyblue', 'blue', 'darkblue',
    'pink', 'lightcoral', 'red', 'darkred'
]

# Configurações do gráfico
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, porcentagens_reducao, color=cores)
ax.set_xlabel('Porcentagem de Redução da Injustiça do Grupo (%)')
ax.set_title('Comparação da Redução da Injustiça do Grupo em relação ao Método Não Federado (%)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(right=max(porcentagens_reducao) * 1.1)

# Adicionando valores nas barras para facilitar a leitura
for bar, value in zip(bars, porcentagens_reducao):
    ax.text(value, bar.get_y() + bar.get_height() / 2, f'{value:.2f}%', va='center', ha='left')

plt.tight_layout()
plt.show()
