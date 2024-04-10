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
    modelos_clientes_perdas = []

    for i, modelo_cliente in enumerate(modelos_clientes):
        # Gerar índices de itens não avaliados
        indices_nao_avaliados = (avaliacoes_inicial[i] == 0).nonzero().squeeze()
        
        if i < 2:
            # Selecionar 2 índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:2]]
            # Gerar 2 novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (2,)).float()
        else:
            # Selecionar 0 índices aleatórios para novas avaliações
            indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:0]]
            # Gerar 0 novas avaliações aleatórias
            novas_avaliacoes = torch.randint(1, 6, (0,)).float()

        # Atualizar avaliações iniciais com novas avaliações
        avaliacoes_cliente = avaliacoes_inicial[i].clone()
        avaliacoes_cliente[indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final[i][indices_novas_avaliacoes] = novas_avaliacoes

        print(f"=== Treinamento no Cliente {i + 1} ===")

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        
        for _ in range(epochs):
            optimizer_cliente.zero_grad()
            output_cliente = modelo_cliente(avaliacoes_cliente.unsqueeze(0))
            loss_cliente = criterion(output_cliente, avaliacoes_cliente.unsqueeze(0))
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(avaliacoes_cliente.unsqueeze(0)).squeeze()
        
        modelos_clientes_perdas.append((i, loss_cliente.item()))

    # Retorna ambos: avaliações finais e os modelos dos clientes
    return avaliacoes_final, modelos_clientes, modelos_clientes_perdas

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

def agregar_modelos_locais_ao_global_pesos_justica_erro(modelo_global, modelos_clientes, modelos_clientes_perdas):
    """
    Atualiza os parâmetros do modelo global com a média ponderada dos parâmetros dos modelos locais,
    baseada nas perdas dos modelos locais.

    Args:
        modelo_global (torch.nn.Module): O modelo global de rede neural.
        modelos_clientes (List[torch.nn.Module]): Lista dos modelos locais treinados.
        modelos_clientes_perdas (List[Tuple[int, float]]): Lista contendo os índices dos modelos e suas perdas.

    Returns:
        List[int]: Índices dos modelos (de 0 a 299) por ordem decrescente de perdas.

    Descrição:
        Esta função calcula as perdas de cada modelo local, e então calcula uma média ponderada
        dos parâmetros do modelo global, dando mais peso aos modelos com maiores perdas.
    """

    # print("perdas")
    # print(modelos_clientes_perdas)

    # Ordenar os modelos com base nas perdas
    modelos_ordenados = sorted(modelos_clientes_perdas, key=lambda x: x[1], reverse=False)

    # Calcular o total de perdas
    total_perdas = sum(perda for _, perda in modelos_clientes_perdas)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_perdas]
    # print("pesos")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

    # Retornar os índices dos modelos por ordem decrescente de perdas
    return [i for i, _ in modelos_ordenados]


def main():
    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO INICIAL) ===")
    caminho_do_arquivo = 'matrix-ratings-u4-i5.csv'
    avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_txt(caminho_do_arquivo)

    numero_de_usuarios = avaliacoes_inicial_tensor.shape[0]
    numero_de_itens = avaliacoes_inicial_tensor.shape[1]

    modelo_global_federado1 = SimpleNN(numero_de_itens, 2, numero_de_itens)
    criterion = nn.MSELoss() 

    treinar_modelo_global(modelo_global_federado1, avaliacoes_inicial_tensor, criterion, 200)
    modelo_global_nao_federado = copy.deepcopy(modelo_global_federado1)
    modelo_global_federado2 = copy.deepcopy(modelo_global_federado1)

    with torch.no_grad():
        recomendacoes_inicial_tensor = modelo_global_federado1(avaliacoes_inicial_tensor)

    print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
    avaliacoes_final_tensor, modelos_clientes, modelos_clientes_perdas = treinar_modelos_locais(modelo_global_federado1, avaliacoes_inicial_tensor, criterion, 200)

    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL - AGREGAÇÃO) ===")
    agregar_modelos_locais_ao_global_pesos(modelo_global_federado1, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes(modelo_global_federado1, modelos_clientes)
    # agregar_modelos_locais_ao_global_pesos_justica(modelo_global_federado1, modelos_clientes)
    # agregar_modelos_locais_ao_global_gradientes_justica(modelo_global_federado1, modelos_clientes)
    list_users = agregar_modelos_locais_ao_global_pesos_justica_erro(modelo_global_federado2, modelos_clientes, modelos_clientes_perdas)

    print("list_users")
    print(list_users)

    print("modelos_clientes_perdas")
    print(modelos_clientes_perdas)

    treinar_modelo_global(modelo_global_nao_federado, avaliacoes_final_tensor, criterion, 200) # Simulando um modelo não federado

    with torch.no_grad():
        recomendacoes_final_tensor1 = modelo_global_federado1(avaliacoes_final_tensor)
        recomendacoes_final_tensor2 = modelo_global_federado2(avaliacoes_final_tensor)
        recomendacoes_modelo_global_nao_federado_tensor = modelo_global_nao_federado(avaliacoes_final_tensor)

    
    print("\n=== MEDIDA DE JUSTIÇA ===")

    avaliacoes_inicial_np = avaliacoes_inicial_tensor.numpy()
    avaliacoes_inicial_df = pd.DataFrame(avaliacoes_inicial_np)
    avaliacoes_final_np = avaliacoes_final_tensor.numpy()
    avaliacoes_final_df = pd.DataFrame(avaliacoes_final_np)
    recomendacoes_inicial_np = recomendacoes_inicial_tensor.numpy()
    recomendacoes_inicial_df = pd.DataFrame(recomendacoes_inicial_np)
    recomendacoes_final_np1 = recomendacoes_final_tensor1.numpy()
    recomendacoes_final_df1 = pd.DataFrame(recomendacoes_final_np1)
    recomendacoes_final_np2 = recomendacoes_final_tensor2.numpy()
    recomendacoes_final_df2 = pd.DataFrame(recomendacoes_final_np2)
    recomendacoes_modelo_global_nao_federado_np = recomendacoes_modelo_global_nao_federado_tensor.numpy()
    recomendacoes_modelo_global_nao_federado_df = pd.DataFrame(recomendacoes_modelo_global_nao_federado_np)
    

    # To capture polarization, we seek to measure the extent to which the user ratings disagree
    polarization = Polarization()
    Rpol_inicial = polarization.evaluate(recomendacoes_inicial_df)
    Rpol_final1 = polarization.evaluate(recomendacoes_final_df1)
    Rpol_final2 = polarization.evaluate(recomendacoes_final_df2)
    Rpol_final_nao_federado = polarization.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f"\nPolarization Inicial (Rpol)               : {Rpol_inicial:.9f}")
    print(f"Polarization Final   (Rpol [1])           : {Rpol_final1:.9f}")
    print(f"Polarization Final   (Rpol [2])           : {Rpol_final2:.9f}")
    print(f"Polarization Final   (Rpol [Não Federado]): {Rpol_final_nao_federado:.9f}")

    # Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
    omega_inicial = (avaliacoes_inicial_df != 0)
    omega_final = (avaliacoes_final_df != 0)

    ilv_inicial = IndividualLossVariance(avaliacoes_inicial_df, omega_inicial, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final1 = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final2 = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_nao_federado = IndividualLossVariance(avaliacoes_final_df, omega_final, 1) #axis = 1 (0 rows e 1 columns)
    Rindv_inicial = ilv_inicial.evaluate(recomendacoes_inicial_df)
    Rindv_final1 = ilv_final1.evaluate(recomendacoes_final_df1)
    Rindv_final2 = ilv_final2.evaluate(recomendacoes_final_df2)
    Rindv_final_nao_federado = ilv_final_nao_federado.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f"\nIndividual Loss Variance (Rindv Inicial)               : {Rindv_inicial:.9f}")
    print(f"Individual Loss Variance (Rindv Final [1])             : {Rindv_final1:.9f}")
    print(f"Individual Loss Variance (Rindv Final [2])             : {Rindv_final2:.9f}")
    print(f"Individual Loss Variance (Rindv Final [Não Federado])  : {Rindv_final_nao_federado:.9f}")

    # # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # # advantaged group: 5% users with the highest number of item ratings
    # # disadvantaged group: 95% users with the lowest number of item ratings
    list_users = recomendacoes_final_df1.index.tolist()
    advantaged_group = list_users[0:2]
    disadvantaged_group = list_users[2:4]
    G = {1: advantaged_group, 2: disadvantaged_group}

    print(avaliacoes_final_df)

    glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final1 = GroupLossVariance(avaliacoes_final_df, omega_final, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final2 = GroupLossVariance(avaliacoes_final_df, omega_final, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_nao_federado = GroupLossVariance(avaliacoes_final_df, omega_final, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_df)
    RgrpNR_final1 = glv_final1.evaluate(recomendacoes_final_df1)
    RgrpNR_final2 = glv_final2.evaluate(recomendacoes_final_df2)
    RgrpNR_final_nao_federado = glv_final_nao_federado.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f"\nGroup Loss Variance (Rgrp Inicial)             : {RgrpNR_inicial:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1])           : {RgrpNR_final1:.9f}")
    print(f"Group Loss Variance (Rgrp Final [2])           : {RgrpNR_final2:.9f}")
    print(f"Group Loss Variance (Rgrp Final [Não Federado]): {RgrpNR_final_nao_federado:.9f}")

    rmse_inicial = RMSE(avaliacoes_inicial_df, omega_inicial)
    result_inicial = rmse_inicial.evaluate(recomendacoes_inicial_df)
    rmse_final1 = RMSE(avaliacoes_final_df, omega_final)
    result_final1 = rmse_final1.evaluate(recomendacoes_final_df1)
    rmse_final2 = RMSE(avaliacoes_final_df, omega_final)
    result_final2 = rmse_final2.evaluate(recomendacoes_final_df2)
    rmse_final_nao_federado = RMSE(avaliacoes_final_df, omega_final)
    result_final_nao_federado = rmse_final_nao_federado.evaluate(recomendacoes_modelo_global_nao_federado_df)
    print(f'\nRMSE Inicial             : {result_inicial:.9f}')
    print(f'RMSE Final [1]           : {result_final1:.9f}')
    print(f'RMSE Final [2]           : {result_final2:.9f}')
    print(f'RMSE Final [Não Federado]: {result_final_nao_federado:.9f}')

    avaliacoes_inicial_df.to_excel("avaliacoes_inicial.xlsx", index=False)
    avaliacoes_final_df.to_excel("avaliacoes_final.xlsx", index=False)
    recomendacoes_inicial_df.to_excel("recomendacoes_inicial.xlsx", index=False)
    recomendacoes_final_df1.to_excel("recomendacoes_final_df1.xlsx", index=False)
    recomendacoes_final_df2.to_excel("recomendacoes_final_df2.xlsx", index=False)
    recomendacoes_modelo_global_nao_federado_df.to_excel("recomendacoes_modelo_global_nao_federado_df.xlsx", index=False)

if __name__ == "__main__":
    main()
