import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance

# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(SimpleNN, self).__init__()
#         self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
#         self.layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
#         self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
#         self.activation = nn.Sigmoid()

#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = torch.relu(layer(x))
#         x = self.activation(self.layers[-1](x)) * 4 + 1  # Scale sigmoid output to range [1, 5]
#         return x

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        
        x = self.activation(self.layers[-1](x))

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
    
def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=50, learning_rate=0.01):
    """
    Treina o modelo global usando uma matriz de avaliações.
    
    Args:
        modelo (torch.nn.Module): O modelo de rede neural a ser treinado.
        avaliacoes (torch.Tensor): Um tensor contendo as avaliações dos usuários sobre os itens.
        criterion (torch.nn.modules.loss._Loss): A função de perda utilizada para o treinamento.
        epochs (int, optional): Número de épocas para o treinamento. Padrão é 50.
        learning_rate (float, optional): Taxa de aprendizado para o otimizador SGD. Padrão é 0.01.
        
    Descrição:
        Esta função treina o modelo global utilizando a matriz de avaliações fornecida.
        Utiliza o otimizador SGD (Descida do Gradiente Estocástica) com a taxa de aprendizado
        especificada. A função de perda calculada a cada época é baseada na diferença entre
        as saídas do modelo e as avaliações reais. Os parâmetros do modelo são atualizados
        em cada passo do treinamento para minimizar a função de perda.
    """
    optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(avaliacoes)
        loss = criterion(output, avaliacoes)
        loss.backward()
        optimizer.step()

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
    avaliacoes_final = avaliacoes_inicial.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes_inicial.size(0))]

    for i, modelo_cliente in enumerate(modelos_clientes):
        # Gerar índices de itens não avaliados
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero().squeeze()
        
        if i < 15:
            # Selecionar 50 índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:50]]
            # Gerar 50 novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (50,)).float()
        else:
            # Selecionar 10 índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:10]]
            # Gerar 10 novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (10,)).float()

        # Atualizar avaliações iniciais com novas avaliações
        avaliacoes_cliente = avaliacoes_inicial[i].clone()
        avaliacoes_cliente[indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final[i][indices_novas_avaliacoes] = novas_avaliacoes

        print(f"\n=== Treinamento no Cliente {i + 1} ===")

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        for _ in range(epochs):
            optimizer_cliente.zero_grad()
            output_cliente = modelo_cliente(avaliacoes_cliente.unsqueeze(0))
            loss_cliente = criterion(output_cliente, avaliacoes_cliente.unsqueeze(0))
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(avaliacoes_cliente.unsqueeze(0)).squeeze()

    # Retorna ambos: avaliações finais e os modelos dos clientes
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
    numero_de_usuarios = avaliacoes_inicial.shape[0]
    numero_de_itens = avaliacoes_inicial.shape[1]

    # model = SimpleNN(input_size, hidden_sizes, output_size)
    modelo_global = SimpleNN(numero_de_itens, [512, 256, 128, 64, 32], numero_de_itens)
    criterion = nn.MSELoss() 

    treinar_modelo_global(modelo_global, avaliacoes_inicial, criterion, 300, 0.005)

    with torch.no_grad():
        recomendacoes_inicial = modelo_global(avaliacoes_inicial)

    print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
    avaliacoes_final, modelos_clientes = treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, 300, 0.005)

    agregar_modelos_locais_ao_global_pesos(modelo_global, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes(modelo_global, modelos_clientes)

    with torch.no_grad():
        recomendacoes_final = modelo_global(avaliacoes_inicial)

    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL) ===")

    
    print("\n\n=== MEDIDA DE PRECISÃO RMSE ===")
    mse_inicial = criterion(recomendacoes_inicial, avaliacoes_inicial).item()
    mse_final = criterion(recomendacoes_final, avaliacoes_final).item()

    print(f"\nMSE Inicial: {mse_inicial:.4f}")
    print(f"MSE Final: {mse_final:.4f}")


    print("\n\n=== MEDIDA DE JUSTIÇA INDIVIDUAL Rindv ===")

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
    # list_users = recomendacoes_final_df.index.tolist()
    # advantaged_group = list_users[0:15]
    # disadvantaged_group = list_users[15:300]
    # G1 = {1: advantaged_group, 2: disadvantaged_group}

    # glv = GroupLossVariance(avaliacoes_inicial_df, omega, G1, 1) #axis = 1 (0 rows e 1 columns)
    # RgrpNR = glv.evaluate(recomendacoes_final_df)
    # print("Group Loss Variance (Rgrp NR - 95-5%):", RgrpNR)

if __name__ == "__main__":
    main()
