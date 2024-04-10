import numpy as np

# Defina as dimensões do arquivo
linhas = 100
colunas = 300

# Gera dados aleatórios - por exemplo, valores inteiros entre 1 e 5
dados = np.random.randint(0, 6, size=(linhas, colunas))

# Salva os dados em um arquivo CSV
np.savetxt('avaliacoes.csv', dados, delimiter=',', fmt='%d')

print("Arquivo 'avaliacoes.csv' gerado com sucesso!")