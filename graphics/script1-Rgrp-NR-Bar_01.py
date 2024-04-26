import matplotlib.pyplot as plt

categorias = [
    'WAVG Fair Loss', 'WAVG Loss', 
    'WAVG Fair Rindv', 'WAVG Rindv',
    'AVG Fair',
    'AVG',
    'NFS'
]

valores = [
    0.197502539, 0.316976041,
    0.176208914, 0.320935562, 
    0.222787817,
    0.298492094,
    0.529993197
]

cores = [
    'lightgreen', 'green',
    'lightblue', 'blue', 
    'pink', 'red', 
    'black'
]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(categorias, valores, color=cores)
ax.set_xlabel('Injustiça do Grupo (NR - Number Ratings)')
ax.set_title('Comparação da Injustiça do Grupo entre Diferentes Métodos de Agregação (Rounds = 5)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(right=max(valores)*1.1)

# Adicionando linhas mais grossas para separar os grupos de 5
ax.axhline(2-0.5, color='black', linewidth=2)
ax.axhline(4-0.5, color='black', linewidth=2)
ax.axhline(6-0.5, color='black', linewidth=2)
ax.axhline(8-0.5, color='black', linewidth=2)

# Adicionando valores nas barras
for bar, value in zip(bars, valores):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.6f}', va='center', ha='left')

plt.tight_layout()
plt.show()
