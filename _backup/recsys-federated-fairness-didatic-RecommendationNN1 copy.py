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
    df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados, df_dados.reset_index(drop=True)

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

        # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, epochs=50, learning_rate=0.01):
    """
    Treina modelos locais para cada conjunto de avaliações de usuário e gera novas avaliações.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        avaliacoes_inicial (torch.Tensor): Tensor contendo as avaliações dos usuários.
        criterion (torch.nn.modules.loss._Loss): A função de perda utilizada para o treinamento.
        epochs (int, optional): Número de épocas para o treinamento. Padrão é 50.
        learning_rate (float, optional): Taxa de aprendizado para o otimizador SGD. Padrão é 0.01.

    Retorna:
        Tuple[torch.Tensor, List[torch.nn.Module]]: Um par contendo o tensor de avaliações finais
        e a lista de modelos locais treinados.

    Descrição:
        Para cada usuário (linha da matriz de avaliações), este método realiza o seguinte processo:
        1. Identifica itens não avaliados pelo usuário.
        2. Seleciona aleatoriamente trinta desses itens para avaliação.
        3. Gera avaliações aleatórias (entre 1 e 5) para esses itens.
        4. Treina um modelo local (uma cópia do modelo global) com as avaliações atualizadas do usuário.
        5. Após o treinamento, gera novas recomendações para o usuário com base no modelo local treinado.
        
        Este processo simula o treinamento local em um cenário de aprendizado federado, onde cada cliente
        atualiza um modelo com base em suas próprias avaliações e, em seguida, o modelo global é atualizado
        com base nos modelos locais.
    """
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

            # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_cliente.item()}')

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

def agregar_modelos_locais_ao_global_pesos_justica(modelo_global, modelos_clientes):
    """
    Atualiza os parâmetros do modelo global com a média ponderada dos parâmetros dos modelos locais.
    Considerando os 300 usuários, temos a seguintes regras:
    Os 15 primeiros usuários (com maiores avaliações) contribuem com 5% da agregação
    Os 285 últimos usuários (com menores avaliações) contribuem com 95% da agregação

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.

    Descrição:
        Esta função percorre cada parâmetro (por exemplo, pesos e vieses) do modelo global e
        atualiza seus valores com a média ponderada dos valores dos parâmetros correspondentes dos
        modelos locais. Essa abordagem permite que os primeiros 15 modelos contribuam com 5%
        na agregação, enquanto os modelos restantes contribuem com 95%.

        O processo de agregação é uma etapa fundamental no aprendizado federado, permitindo
        que o modelo global beneficie-se do aprendizado distribuído sem a necessidade de
        compartilhar diretamente os dados locais, preservando assim a privacidade dos dados.

        É importante que esta função seja chamada após o treinamento dos modelos locais e
        antes da próxima rodada de distribuição do modelo global atualizado para treinamento
        local adicional ou para inferência.
    """
    num_clientes = len(modelos_clientes)
    peso_clientes = [0.05] * min(num_clientes, 15)  # Primeiros 15 modelos contribuem com 5%
    peso_restante = 0.95  # Peso para os modelos restantes

    if num_clientes > 15:
        peso_restante /= num_clientes - 15  # Distribui o restante igualmente entre os modelos restantes
        peso_clientes.extend([peso_restante] * (num_clientes - 15))

    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)  # Inicializa o tensor para armazenar a média

            # Calcular a média dos parâmetros de cada cliente
            for peso, cliente in zip(peso_clientes, modelos_clientes):
                cliente_params = list(cliente.parameters())[i].data  # Parâmetros do cliente atual
                param_medio += peso * cliente_params  # Adiciona à média ponderada

            # Atualizar os parâmetros globais com a média ponderada
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_gradientes_justica(modelo_global, modelos_clientes, learning_rate=0.01):
    """
    Atualiza os parâmetros do modelo global com a média ponderada dos gradientes dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.
        learning_rate (float): Taxa de aprendizado para a atualização dos parâmetros do modelo global.

    Descrição:
        Esta função percorre cada parâmetro do modelo global e atualiza seus valores com a média ponderada dos gradientes
        dos parâmetros correspondentes dos modelos locais. A ponderação é determinada pela regra de justiça, onde os
        primeiros 15 modelos contribuem com 5% da agregação, enquanto os modelos restantes contribuem com 95%.

        Este método é uma abordagem alternativa ao processo padrão de agregação em aprendizado federado, permitindo um
        ajuste mais fino do modelo global com base nas tendências de aprendizado locais e considerando justiça na agregação.

        É importante ressaltar que, para que esta função funcione como esperado, os modelos locais devem ter seus gradientes
        retidos (não zerados) após o treinamento.
    """
    num_clientes = len(modelos_clientes)
    peso_clientes = [0.05] * min(num_clientes, 15)  # Primeiros 15 modelos contribuem com 5%
    peso_restante = 0.95  # Peso para os modelos restantes

    if num_clientes > 15:
        peso_restante /= num_clientes - 15  # Distribui o restante igualmente entre os modelos restantes
        peso_clientes.extend([peso_restante] * (num_clientes - 15))

    with torch.no_grad():
        global_params = list(modelo_global.parameters())
        
        # Inicializar uma lista para armazenar a média ponderada dos gradientes para cada parâmetro
        gradientes_medios = [torch.zeros_like(param) for param in global_params]
        
        # Calcular a média ponderada dos gradientes para cada parâmetro
        for peso, modelo_cliente in zip(peso_clientes, modelos_clientes):
            for i, param_cliente in enumerate(modelo_cliente.parameters()):
                if param_cliente.grad is not None:
                    gradientes_medios[i] += peso * param_cliente.grad
        
        # Atualizar os parâmetros do modelo global usando a média ponderada dos gradientes
        for i, param_global in enumerate(global_params):
            param_global -= learning_rate * gradientes_medios[i]


def main():
    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO INICIAL) ===")
    caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
    avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

    with torch.no_grad():
        # Preparar os índices de usuários e itens
        num_users, num_items = avaliacoes_inicial_tensor.shape
        user_ids, item_ids = torch.meshgrid(torch.arange(num_users), torch.arange(num_items), indexing='ij')
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
    optimizer = optim.Adam(modelo_global.parameters(), lr=0.001, weight_decay=0.001)  # Adam como otimizador com regularização L2

    treinar_modelo_global(modelo_global, user_ids, item_ids, avaliacoes_inicial_tensor, criterion, 300, 0.005)

    with torch.no_grad():
        recomendacoes_inicial_tensor = modelo_global(user_ids_long, item_ids_long).view(num_users, num_items)

    print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
    avaliacoes_final_tensor, modelos_clientes = treinar_modelos_locais(modelo_global, avaliacoes_inicial_tensor, criterion, 300, 0.005)

    # agregar_modelos_locais_ao_global_pesos(modelo_global, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes(modelo_global, modelos_clientes)
    agregar_modelos_locais_ao_global_pesos_justica(modelo_global, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes_justica(modelo_global, modelos_clientes)

    with torch.no_grad():
        recomendacoes_final_tensor = modelo_global(user_ids_long, item_ids_long).view(num_users, num_items)

    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL) ===")
    
    print("\n\n=== MEDIDA DE PRECISÃO RMSE ===")
    mse_inicial = criterion(recomendacoes_inicial_tensor, avaliacoes_inicial_tensor).item()
    mse_final = criterion(recomendacoes_final_tensor, avaliacoes_final_tensor).item()

    print(f"MSE Inicial: {mse_inicial:.4f}")
    print(f"MSE Final: {mse_final:.4f}")


    print("\n=== MEDIDA DE JUSTIÇA ===")

    avaliacoes_inicial_np = avaliacoes_inicial_tensor.numpy()
    avaliacoes_inicial_df = pd.DataFrame(avaliacoes_inicial_np)
    avaliacoes_final_np = avaliacoes_final_tensor.numpy()
    avaliacoes_final_df = pd.DataFrame(avaliacoes_final_np)
    recomendacoes_inicial_np = recomendacoes_inicial_tensor.numpy()
    recomendacoes_inicial_df = pd.DataFrame(recomendacoes_inicial_np)
    recomendacoes_final_np = recomendacoes_final_tensor.numpy()
    recomendacoes_final_df = pd.DataFrame(recomendacoes_final_np)

    # To capture polarization, we seek to measure the extent to which the user ratings disagree
    polarization = Polarization()
    Rpol_inicial = polarization.evaluate(recomendacoes_inicial_df)
    Rpol_final = polarization.evaluate(recomendacoes_final_df)
    print(f"Polarization Inicial (Rpol): {Rpol_inicial:.9f}")
    print(f"Polarization Final   (Rpol): {Rpol_final:.9f}")

    # Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
    omega_inicial = (avaliacoes_inicial_df != 0)
    omega_final = (avaliacoes_final_df != 0)

    ilv_inicial = IndividualLossVariance(avaliacoes_inicial_df, omega_inicial, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    Rindv_inicial = ilv_inicial.evaluate(recomendacoes_inicial_df)
    Rindv_final = ilv_final.evaluate(recomendacoes_final_df)
    print(f"Individual Loss Variance (Rindv Inicial): {Rindv_inicial:.9f}")
    print(f"Individual Loss Variance (Rindv Final)  : {Rindv_final:.9f}")

    # # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # # advantaged group: 5% users with the highest number of item ratings
    # # disadvantaged group: 95% users with the lowest number of item ratings
    list_users = recomendacoes_final_df.index.tolist()
    advantaged_group = list_users[0:15]
    disadvantaged_group = list_users[15:300]
    G = {1: advantaged_group, 2: disadvantaged_group}

    glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final = GroupLossVariance(avaliacoes_final_df, omega_final, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_df)
    RgrpNR_final = glv_final.evaluate(recomendacoes_final_df)
    print(f"Group Loss Variance (Rgrp Inicial NR - 95-5%): {RgrpNR_inicial:.9f}")
    print(f"Group Loss Variance (Rgrp Final NR - 95-5%)  : {RgrpNR_final:.9f}")

    rmse_inicial = RMSE(avaliacoes_inicial_df, omega_inicial)
    result_inicial = rmse_inicial.evaluate(recomendacoes_inicial_df)
    rmse_final = RMSE(avaliacoes_final_df, omega_final)
    result_final = rmse_final.evaluate(recomendacoes_final_df)
    print(f'RMSE Inicial: {result_inicial:.9f}')
    print(f'RMSE Final: {result_final:.9f}\n')

    avaliacoes_inicial_df.to_excel("avaliacoes_inicial.xlsx", index=False)
    avaliacoes_final_df.to_excel("avaliacoes_final.xlsx", index=False)
    recomendacoes_inicial_df.to_excel("recomendacoes_inicial.xlsx", index=False)
    recomendacoes_final_df.to_excel("recomendacoes_final.xlsx", index=False)

if __name__ == "__main__":
    main()
