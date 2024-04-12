import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd


class RecommendationNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_size):
        super(RecommendationNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(2 * embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        x = torch.cat((user_embedded, item_embedded), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Usando sigmoid para garantir valores entre 0 e 1
        # Transformando os valores para o intervalo desejado (1 a 5)
        x = 4 * x + 1
        return x.view(-1)


def carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo):
    df = pd.read_excel(caminho_do_arquivo) # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df_com_zero = df.fillna(0) # Substituir valores NaN por zero
    df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados, df_dados.reset_index(drop=True)


def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=50, learning_rate=0.1):
    optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)
    num_usuarios, num_itens = avaliacoes.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten(), itens_ids.flatten()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = modelo(usuarios_ids.long(), itens_ids.long()).view(num_usuarios, num_itens)
        loss = criterion(predictions, avaliacoes.float())
        loss.backward()
        optimizer.step()



import itertools

caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)
num_usuarios, num_itens = avaliacoes_inicial_tensor.shape
usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
usuarios_ids = usuarios_ids.reshape(-1)
itens_ids = itens_ids.reshape(-1)
usuarios_ids_long = usuarios_ids.long()
itens_ids_long = itens_ids.long()

# Exemplo de uso
# num_usuarios = len(usuarios_ids_long)
# num_itens = len(itens_ids_long)
embedding_size = 64
hidden_size = 128


# Defina os valores para a pesquisa em grade
# learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
# epochs_list = [200, 250, 300, 350, 400]
learning_rates = [0.001, 0.002, 0.003, 0.008, 0.010, 0.015, 0.020]
epochs_list = [350, 400]
# Melhores hiperparâmetros: {'learning_rate': 0.005, 'epochs': 300}
# Melhores hiperparâmetros: {'learning_rate': 0.02, 'epochs': 400}
# Adicione mais hiperparâmetros conforme necessário

best_loss = float('inf')
best_hyperparams = {}

# Realize a pesquisa em grade
for learning_rate, epochs in itertools.product(learning_rates, epochs_list):
    print(f"learning_rate : {learning_rate}, epochs : {epochs}")
    modelo_global_federado1 = RecommendationNN(num_usuarios, num_itens, embedding_size, hidden_size)
    optimizer = optim.SGD(modelo_global_federado1.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        treinar_modelo_global(modelo_global_federado1, avaliacoes_inicial_tensor, criterion, 1, learning_rate)

    with torch.no_grad():
        predictions_val = modelo_global_federado1(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        loss_val = criterion(predictions_val, avaliacoes_inicial_tensor.float())

    # Se encontrou um modelo melhor, atualize os melhores parâmetros
    if loss_val < best_loss:
        best_loss = loss_val
        best_hyperparams = {'learning_rate': learning_rate, 'epochs': epochs}  # Adicione mais hiperparâmetros conforme necessário

# Imprima os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", best_hyperparams)






