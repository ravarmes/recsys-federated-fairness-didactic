import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance

class RecommendationNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64, hidden_size=128):
        super(RecommendationNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        x = torch.cat((user_embed, item_embed), dim=1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
def carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo):
    df = pd.read_excel(caminho_do_arquivo) # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df_com_zero = df.fillna(0) # Substituir valores NaN por zero
    dados_filmes = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(dados_filmes.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados

def carregar_avaliacoes_do_arquivo_txt(caminho_do_arquivo):
    dados = np.loadtxt(caminho_do_arquivo, delimiter=',', dtype=np.float32)
    return torch.tensor(dados)
    
def treinar_modelo_global(modelo, user_ids, item_ids, avaliacoes, criterion, epochs=50, learning_rate=0.01):
    optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)

    modelo.train()  # Coloca o modelo em modo de treinamento
    num_usuarios, num_itens = avaliacoes.shape

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Converte user_ids e item_ids para o tipo Long corretamente antes de passá-los como entrada para o modelo.
        user_ids_long = user_ids.long()
        item_ids_long = item_ids.long()

        # Realiza as previsões
        predictions = modelo(user_ids_long, item_ids_long)  # Use as versões convertidas
        predictions = predictions.squeeze()  # Remove qualquer dimensão extra.

        # Redimensiona as previsões para coincidir com a forma da matriz de avaliações
        predictions_reshaped = predictions.view(num_usuarios, num_itens)

        loss = criterion(predictions_reshaped, avaliacoes.float())  # Garante que as avaliações também estejam no tipo correto

        loss.backward()  # Backpropagation
        optimizer.step()  # Atualização dos pesos

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, epochs=50, learning_rate=0.01):
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes_inicial.size(0))]
    avaliacoes_final = avaliacoes_inicial.clone()

    num_usuarios, num_itens = avaliacoes_inicial.shape
    user_indices_total, item_indices_total = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    
    for i, modelo_cliente in enumerate(modelos_clientes):
        print(f"\n=== Treinamento no Cliente {i + 1} ===")
        
        # Identificar itens não avaliados pelo usuário
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero().squeeze()

        # Selecionar índices aleatórios com base no critério original e gerar novas avaliações
        if i < 15:
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:120]]
            novas_avaliacoes = torch.randint(1, 6, (120,)).float()
        else:
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:10]]
            novas_avaliacoes = torch.randint(1, 6, (10,)).float()

        # Atualizar a matriz de avaliações inicial com as novas avaliações
        avaliacoes_final[i][indices_novas_avaliacoes] = novas_avaliacoes

        # Para o treinamento, agora usaremos a matriz de avaliações finais com as novas avaliações incluídas
        usuario_id = torch.tensor([i] * num_itens)
        avaliacoes_usuario = avaliacoes_final[i]

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        modelo_cliente.train()

        for epoch in range(epochs):
            optimizer_cliente.zero_grad()

            usuario_id_long = usuario_id.long()
            item_ids_long = item_indices_total[i].long()

            output_cliente = modelo_cliente(usuario_id_long, item_ids_long).squeeze()
            loss_cliente = criterion(output_cliente, avaliacoes_usuario.float())
            loss_cliente.backward()
            optimizer_cliente.step()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_cliente.item()}')

        # Aqui, poderíamos atualizar as avaliações finais com as previsões do modelo local para todos os itens,
        # Mas já fizemos isso ao adicionar as novas avaliações aleatórias

    # Retorna tanto as avaliações finais quanto os modelos dos clientes.
    return avaliacoes_final, modelos_clientes

def agregar_modelos_locais_ao_global_pesos(modelo_global, modelos_clientes):
    """
    Atualiza os parâmetros do modelo global com a média dos parâmetros dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.

    Descrição:
        Esta função percorre cada parâmetro (por exemplo, pesos e vieses) do modelo global e
        atualiza seus valores com a média dos valores dos parâmetros correspondentes dos
        modelos locais. Essa abordagem assume que uma média simples dos parâmetros pode
        levar a um modelo global mais genérico e robusto, incorporando o aprendizado obtido
        de várias fontes de dados locais.

        O processo de agregação é uma etapa fundamental no aprendizado federado, permitindo
        que o modelo global beneficie-se do aprendizado distribuído sem a necessidade de
        compartilhar diretamente os dados locais, preservando assim a privacidade dos dados.

        É importante que esta função seja chamada após o treinamento dos modelos locais e
        antes da próxima rodada de distribuição do modelo global atualizado para treinamento
        local adicional ou para inferência.
    """
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
            param_global.copy_(cliente_params.mean(dim=0))

def agregar_modelos_locais_ao_global_gradientes(modelo_global, modelos_clientes, learning_rate=0.01):
    """
    Atualiza os parâmetros do modelo global com a média dos gradientes dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.

    Descrição:
        Esta função percorre cada parâmetro (por exemplo, pesos e vieses) do modelo global e
        atualiza seus valores com a média dos gradientes dos parâmetros correspondentes dos
        modelos locais. A ideia é que, em vez de atualizar o modelo global com a média direta
        dos parâmetros dos modelos locais, utilizamos os gradientes (derivadas da função de
        perda em relação aos parâmetros) para fazer uma atualização baseada em como cada
        modelo local "aprendeu" a partir de seus dados.

        Este método é uma abordagem alternativa ao processo padrão de agregação em
        aprendizado federado, permitindo um ajuste mais fino do modelo global com base nas
        tendências de aprendizado locais.

        É importante ressaltar que, para que esta função funcione como esperado, os modelos
        locais devem ter seus gradientes retidos (não zerados) após o treinamento.
    """
    with torch.no_grad():
        global_params = list(modelo_global.parameters())
        
        # Inicializar uma lista para armazenar a média dos gradientes para cada parâmetro
        gradientes_medios = [torch.zeros_like(param) for param in global_params]
        
        # Calcular a média dos gradientes para cada parâmetro
        for modelo_cliente in modelos_clientes:
            for i, param_cliente in enumerate(modelo_cliente.parameters()):
                if param_cliente.grad is not None:
                    gradientes_medios[i] += param_cliente.grad / len(modelos_clientes)
        
        # Atualizar os parâmetros do modelo global usando a média dos gradientes
        for i, param_global in enumerate(global_params):
            param_global -= learning_rate * gradientes_medios[i]


def main():
    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO INICIAL) ===")
    caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
    avaliacoes_inicial = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

    with torch.no_grad():
        # Preparar os índices de usuários e itens
        num_usuarios, num_itens = avaliacoes_inicial.shape
        user_ids, item_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
        user_ids = user_ids.reshape(-1)
        item_ids = item_ids.reshape(-1)

        # Assegurando que os índices sejam inteiros longos para compatibilidade com embedding layers
        user_ids_long = user_ids.long()
        item_ids_long = item_ids.long()

    # Exemplo de uso
    num_users = 300
    num_items = 1000
    embedding_size = 64
    hidden_size = 128

    modelo_global = RecommendationNN(num_users, num_items, embedding_size, hidden_size)
    criterion = nn.MSELoss() 

    treinar_modelo_global(modelo_global, user_ids, item_ids, avaliacoes_inicial, criterion, 300, 0.005)

    with torch.no_grad():
        recomendacoes_inicial = modelo_global(user_ids_long, item_ids_long).view(num_usuarios, num_itens)

    print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
    avaliacoes_final, modelos_clientes = treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, 300, 0.005)

    agregar_modelos_locais_ao_global_pesos(modelo_global, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes(modelo_global, modelos_clientes)

    with torch.no_grad():
        recomendacoes_final = modelo_global(user_ids_long, item_ids_long).view(num_usuarios, num_itens)

    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL) ===")

    
    print("\n\n=== MEDIDA DE PRECISÃO RMSE ===")
    mse_inicial = criterion(recomendacoes_inicial, avaliacoes_inicial).item()
    mse_final = criterion(recomendacoes_final, avaliacoes_final).item()

    print(f"MSE Inicial: {mse_inicial:.4f}")
    print(f"MSE Final: {mse_final:.4f}")


    print("\n=== MEDIDA DE JUSTIÇA INDIVIDUAL Rindv ===")

    avaliacoes_inicial_df = pd.DataFrame(avaliacoes_inicial)
    avaliacoes_final_df = pd.DataFrame(avaliacoes_final)
    recomendacoes_inicial_df = pd.DataFrame(recomendacoes_inicial)
    recomendacoes_final_df = pd.DataFrame(recomendacoes_final)

    # To capture polarization, we seek to measure the extent to which the user ratings disagree
    polarization = Polarization()
    Rpol_inicial = polarization.evaluate(recomendacoes_inicial_df)
    Rpol_final = polarization.evaluate(recomendacoes_final_df)
    print("Polarization Inicial (Rpol):", Rpol_inicial)
    print("Polarization Final   (Rpol):", Rpol_final)

    # Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
    omega_inicial = ~avaliacoes_inicial_df.isnull()
    omega_final = ~avaliacoes_final_df.isnull()
    ilv_inicial = IndividualLossVariance(avaliacoes_inicial_df, omega_inicial, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    Rindv_inicial = ilv_inicial.evaluate(recomendacoes_inicial_df)
    Rindv_final = ilv_final.evaluate(recomendacoes_final_df)
    print("Individual Loss Variance (Rindv Inicial):", Rindv_inicial)
    print("Individual Loss Variance (Rindv Final):", Rindv_final)

    # # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # # advantaged group: 5% users with the highest number of item ratings
    # # disadvantaged group: 95% users with the lowest number of item ratings
    list_users = recomendacoes_final_df.index.tolist()
    advantaged_group = list_users[0:15]
    disadvantaged_group = list_users[15:300]
    G = {1: advantaged_group, 2: disadvantaged_group}

    glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final = GroupLossVariance(avaliacoes_inicial_df, omega_final, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_df)
    RgrpNR_final = glv_final.evaluate(recomendacoes_final_df)
    print("Group Loss Variance (Rgrp Inicial NR - 95-5%):", RgrpNR_inicial)
    print("Group Loss Variance (Rgrp Final NR - 95-5%):", RgrpNR_final)

if __name__ == "__main__":
    main()
