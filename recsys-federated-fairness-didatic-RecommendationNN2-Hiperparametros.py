from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Defina sua classe de rede neural aqui
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

# Defina a função para carregar os dados aqui
def carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo):
    df = pd.read_excel(caminho_do_arquivo) # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df_com_zero = df.fillna(0) # Substituir valores NaN por zero
    df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados, df_dados.reset_index(drop=True)

# Carregue os dados
caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

# Crie o modelo
num_users = 300
num_items = 1000
embedding_size = 64
hidden_size = 128
modelo = RecommendationNN(num_users, num_items, embedding_size, hidden_size)

# Defina os hiperparâmetros para pesquisa
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'weight_decay': [0.001, 0.01, 0.1],
    'embedding_size': [32, 64, 128],
    'hidden_size': [64, 128, 256]
}

# Defina a função de perda e o otimizador
criterion = nn.MSELoss()

# Defina o número de épocas
num_epochs = 10

# Execute a pesquisa de hiperparâmetros
grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(avaliacoes_inicial_tensor, avaliacoes_inicial_tensor)  # Substitua os dados de treinamento aqui

# Imprima os melhores parâmetros encontrados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)
