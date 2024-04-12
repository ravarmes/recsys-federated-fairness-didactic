import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.activation(self.layer2(x)) * 4 + 1  # Scale sigmoid output to range [1, 5]
        return x
    
def carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo):
    df = pd.read_excel(caminho_do_arquivo) # Carregar os dados do arquivo Excel para um DataFrame do pandas
    df_com_zero = df.fillna(0) # Substituir valores NaN por zero
    df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
    tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
    return tensor_dados, df_dados.reset_index(drop=True)

def carregar_avaliacoes_do_arquivo_txt(caminho_do_arquivo):
    dados = np.loadtxt(caminho_do_arquivo, delimiter=',', dtype=np.float32)
    return torch.tensor(dados), dados
    
def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=200, learning_rate=0.01):
    # optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)
    optimizer = optim.Adam(modelo.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(avaliacoes)
        loss = criterion(output, avaliacoes)
        loss.backward()
        optimizer.step()

def treinar_modelos_locais(modelo_global, avaliacoes_inicial, criterion, epochs=200, learning_rate=0.01):
    """
    Treina modelos locais para cada conjunto de avaliações de usuário e gera novas avaliações.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        avaliacoes_inicial (torch.Tensor): Tensor contendo as avaliações dos usuários.
        criterion (torch.nn.modules.loss._Loss): A função de perda utilizada para o treinamento.
        epochs (int, optional): Número de épocas para o treinamento. Padrão é 50.
        learning_rate (float, optional): Taxa de aprendizado para o otimizador SGD. Padrão é 0.01.

    Retorna:
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.nn.Module], List[Tuple[int, float]]]: Um par contendo o tensor de avaliações finais,
        a lista de tensores de avaliações finais por cliente, a lista de modelos locais treinados e a lista de perdas dos modelos locais.

    Descrição:
        Este método treina modelos locais para cada cliente com base em todas as avaliações disponíveis,
        incluindo as novas avaliações realizadas localmente. Em seguida, gera novas recomendações para
        cada cliente com base nos modelos locais treinados.
    """
    avaliacoes_final = avaliacoes_inicial.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes_inicial.size(0))]
    modelos_clientes_rindv = [] # injustiças individuais de cada cliente local em seu respectivo modelo local
    modelos_clientes_loss = [] # perdas dos modelos locais de cada cliente local
    modelos_clientes_nr = [] # número de avaliações de cada cliente local

    NR_ADVANTAGED_GROUP = 120      # número de avaliações geradas para os clientes do grupo dos favorecidos
    NR_DISADVANTAGED_GROUP = 10  # número de avaliações geradas para os clientes do grupo dos desfavorecidos

    for i, modelo_cliente in enumerate(modelos_clientes):
        # Gerar índices de itens não avaliados
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero().squeeze()
        
        if i < 15:
            # Selecionar NR_ADVANTAGED_GROUP índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP]]
            # Gerar NR_ADVANTAGED_GROUP novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (NR_ADVANTAGED_GROUP,)).float()
            modelos_clientes_nr.append((i, NR_ADVANTAGED_GROUP))
        else:
            # Selecionar NR_DISADVANTAGED_GROUP índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_DISADVANTAGED_GROUP]]
            # Gerar NR_DISADVANTAGED_GROUP novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (NR_DISADVANTAGED_GROUP,)).float()
            modelos_clientes_nr.append((i, NR_DISADVANTAGED_GROUP))

        # Atualizar avaliações iniciais com novas avaliações
        avaliacoes_final[i][indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final_cliente = avaliacoes_inicial.clone()  # Usar clone para manter as avaliações iniciais
        avaliacoes_final_cliente[i][indices_novas_avaliacoes] = novas_avaliacoes

        print(f"=== Treinamento no Cliente {i + 1} ===")

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        
        for _ in range(epochs):
            optimizer_cliente.zero_grad()
            output_cliente = modelo_cliente(avaliacoes_final_cliente.unsqueeze(0))
            loss_cliente = criterion(output_cliente, avaliacoes_final_cliente.unsqueeze(0))
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(avaliacoes_final_cliente.unsqueeze(0)).squeeze()
        
        # Calculando as perdas individuais (lis) dos clientes em cada modelo de cliente
        avaliacoes_final_cliente_np = avaliacoes_final_cliente.numpy()
        avaliacoes_final_cliente_df = pd.DataFrame(avaliacoes_final_cliente_np)
        recomendacoes_cliente_np = recomendacoes_cliente.numpy()
        recomendacoes_cliente_df = pd.DataFrame(recomendacoes_cliente_np)
        omega_avaliacoes_final_cliente_df = (avaliacoes_final_cliente_df != 0)

        ilv_cliente = IndividualLossVariance(avaliacoes_final_cliente_df, omega_avaliacoes_final_cliente_df, 1)
        lis_cliente = ilv_cliente.get_losses(recomendacoes_cliente_df)

        modelos_clientes_rindv.append((i, lis_cliente[i])) # injustiças individuais do cliente local em seu respectivo modelo local
        modelos_clientes_loss.append((i, loss_cliente.item())) # perdas dos modelos locais


    # Retorna ambos: avaliações finais, avaliações finais por cliente, modelos dos clientes e perdas dos modelos (baseado na injustiça individual li)
    return avaliacoes_final, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr

def agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global, modelos_clientes):
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

def agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global, modelos_clientes, learning_rate=0.01):
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

def agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global, modelos_clientes, modelos_clientes_rindv):
    # Calcular o total de perdas
    total_perdas = sum(perda for _, perda in modelos_clientes_rindv)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_rindv]
    print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_rindv")
    print("modelos_clientes_rindv")
    print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global, modelos_clientes, modelos_clientes_loss):
    # Calcular o total de perdas
    total_perdas = sum(perda for _, perda in modelos_clientes_loss)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_loss]
    print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_loss")
    print("modelos_clientes_loss")
    print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global, modelos_clientes, modelos_clientes_nr):
    # Calcular o total_nr
    total_nr = sum(nr for _, nr in modelos_clientes_nr)

    # Calcular os pesos de agregação baseados nos valores inversos de total_nr
    pesos = [total_nr / nr if nr != 0 else 0 for _, nr in modelos_clientes_nr]

    # Normalizar os pesos para que a soma seja 1
    total_pesos = sum(pesos)
    pesos = [peso / total_pesos for peso in pesos]
    print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_nr")
    print("modelos_clientes_nr")
    print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

import itertools

caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

numero_de_usuarios = avaliacoes_inicial_tensor.shape[0]
numero_de_itens = avaliacoes_inicial_tensor.shape[1]

# Defina os valores para a pesquisa em grade
learning_rates = [0.001, 0.005, 0.010, 0.015, 0.020, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040]
epochs_list = [100, 300, 500, 700, 1000, 1200, 1500, 1600, 1700, 1800, 1900, 2000]
# Melhores hiperparâmetros: {'learning_rate': 0.031, 'epochs': 1500}
# Melhores hiperparâmetros: {'learning_rate': 0.035, 'epochs': 2000}
# Adicione mais hiperparâmetros conforme necessário

best_loss = float('inf')
best_hyperparams = {}

# Realize a pesquisa em grade
for learning_rate, epochs in itertools.product(learning_rates, epochs_list):
    print(f"learning_rate : {learning_rate}, epochs : {epochs}")
    modelo_global_federado1 = SimpleNN(numero_de_itens, 20, numero_de_itens)
    optimizer = optim.SGD(modelo_global_federado1.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        treinar_modelo_global(modelo_global_federado1, avaliacoes_inicial_tensor, criterion, 1, learning_rate)

    with torch.no_grad():
        predictions_val = modelo_global_federado1(avaliacoes_inicial_tensor)
        loss_val = criterion(predictions_val, avaliacoes_inicial_tensor.float())

    # Se encontrou um modelo melhor, atualize os melhores parâmetros
    if loss_val < best_loss:
        best_loss = loss_val
        best_hyperparams = {'learning_rate': learning_rate, 'epochs': epochs}  # Adicione mais hiperparâmetros conforme necessário

# Imprima os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", best_hyperparams)