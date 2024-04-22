import matplotlib.pyplot as plt

# Dados
categorias = [
    'WAVG Fair NR', 'WAVG NR', 
    'WAVG Fair Loss', 'WAVG Loss', 
    'WAVG Fair Rindv', 'WAVG Rindv',
    'AVG Fair',
    'AVG',
    'NFS'
]

valores = [
    0.253794730, 0.333889246,
    0.191555917, 0.346457183,
    0.176923722, 0.350863844, 
    (0.201010257 + 0.201281041 + 0.209656239) / 3,
    (0.346341550 + 0.266793668 + 0.296403408) / 3,
    (0.483585000 + 0.480675876 + 0.483972609) / 3,
]

# Criando gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(categorias, valores, color='skyblue')
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Comparação entre Categorias')
plt.xticks(rotation=45, ha='right')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(categorias, valores, marker='o', linestyle='-', color='b')
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Variação entre Categorias')
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(categorias, valores, color='red', s=50)  # s controla o tamanho dos pontos
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Dispersão dos Valores entre Categorias')
plt.xticks(rotation=45, ha='right')
plt.show()