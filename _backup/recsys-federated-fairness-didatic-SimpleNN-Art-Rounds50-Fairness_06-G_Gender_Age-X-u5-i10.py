import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
import random
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance
from AlgorithmImpartiality import AlgorithmImpartiality

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
    
def treinar_modelo_global(modelo, avaliacoes, criterion, epochs=5, learning_rate=0.02):
    optimizer = optim.SGD(modelo.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(modelo.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    modelo.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(avaliacoes)
        loss = criterion(output, avaliacoes)
        loss.backward()
        optimizer.step()

def treinar_modelos_locais(modelo_global, avaliacoes, avaliacoes_random, G, criterion, epochs=5, learning_rate=0.02):
    avaliacoes_final = avaliacoes.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes.size(0))]
    modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr = [], [], []
    
    NUMBER_ADVANTAGED_GROUP = 2
    NR_ADVANTAGED_GROUP = 1
    NR_DISADVANTAGED_GROUP = 1

    avaliacoes_random_tensor = torch.tensor(avaliacoes_random)
    avaliacoes_random_tensor = avaliacoes_random_tensor.float()

    for i, modelo_cliente in enumerate(modelos_clientes):
        indices_nao_avaliados = (avaliacoes[i] == 0).nonzero(as_tuple=False).squeeze()
        indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP]]
        novas_avaliacoes = avaliacoes_random_tensor[0:NR_ADVANTAGED_GROUP] if i < NUMBER_ADVANTAGED_GROUP else avaliacoes_random_tensor[0:NR_DISADVANTAGED_GROUP]

        avaliacoes_final_cliente = avaliacoes_final.clone()
        avaliacoes_final_cliente[i][indices_novas_avaliacoes] = novas_avaliacoes

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate)
        modelo_cliente.train()

        for _ in range(epochs):
            optimizer_cliente.zero_grad()
            output_cliente = modelo_cliente(avaliacoes_final_cliente[i])
            loss_cliente = criterion(output_cliente, avaliacoes_final_cliente[i])
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(avaliacoes_final_cliente)

        avaliacoes_final[i] = avaliacoes_final_cliente[i]

        avaliacoes_final_cliente_np = avaliacoes_final_cliente.numpy()
        avaliacoes_final_cliente_df = pd.DataFrame(avaliacoes_final_cliente_np)
        recomendacoes_cliente_np = recomendacoes_cliente.numpy()
        recomendacoes_cliente_df = pd.DataFrame(recomendacoes_cliente_np)
        omega_avaliacoes_final_cliente_df = (avaliacoes_final_cliente_df != 0)

        ilv_cliente = IndividualLossVariance(avaliacoes_final_cliente_df, omega_avaliacoes_final_cliente_df, 1)
        lis_cliente = ilv_cliente.get_losses(recomendacoes_cliente_df)

        modelos_clientes_rindv.append((i, lis_cliente[i]))
        modelos_clientes_loss.append((i, loss_cliente.item()))

        quantidade_valores_diferentes_de_zero = len([valor for valor in avaliacoes_final_cliente[i] if valor != 0])
        modelos_clientes_nr.append((i, quantidade_valores_diferentes_de_zero))

    return avaliacoes_final, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr


def agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global, modelos_clientes):
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
            param_global.copy_(cliente_params.mean(dim=0))

def agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global, modelos_clientes, modelos_clientes_rindv):
    # Calcular o total das injustiças individuais (Rindv)
    total_rindv = sum(rindv for _, rindv in modelos_clientes_rindv)

    # Calcular os pesos de agregação baseados nas rindv's
    pesos = [rindv / total_rindv for _, rindv in modelos_clientes_rindv]

    # print("\n\agregar_modelos_locais_ao_global_media_poderada_pesos_rindv")
    # print("modelos_clientes_rindv :: pesos")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global, modelos_clientes, modelos_clientes_loss):
    # Calcular o total de perdas dos modelos locais (loss)
    total_perdas = sum(perda for _, perda in modelos_clientes_loss)

    # Calcular os pesos de agregação baseados nas perdas (loss)
    pesos = [perda / total_perdas for _, perda in modelos_clientes_loss]
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_loss")
    # print("modelos_clientes_loss :: pesos")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global, modelos_clientes, modelos_clientes_nr):
    # Calcular o número total de avaliações (NR - Number Ratings)
    total_nr = sum(nr for _, nr in modelos_clientes_nr)
    # print("total_nr")
    # print(total_nr)

    # Calcular os pesos de agregação baseados nos valores inversos de total_nr
    pesos = [total_nr / nr if nr != 0 else 0 for _, nr in modelos_clientes_nr]

    # Normalizar os pesos para que a soma seja 1
    total_pesos = sum(pesos)
    pesos = [peso / total_pesos for peso in pesos]
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_nr")
    # print("modelos_clientes_nr :: pesos")
    # print(pesos)

    # Atualizar os parâmetros do modelo global com a média ponderada
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            param_medio = torch.zeros_like(param_global)
            for j, peso in enumerate(pesos):
                cliente_params = list(modelos_clientes[j].parameters())[i].data
                param_medio += peso * cliente_params
            param_global.copy_(param_medio)

def aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global, avaliacoes, G):
    
    avaliacoes_np = avaliacoes.numpy()
    avaliacoes_df = pd.DataFrame(avaliacoes_np)

    omega = (avaliacoes_df != 0)

    num_usuarios, num_itens = avaliacoes.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten(), itens_ids.flatten()
    with torch.no_grad():
        recomendacoes_tensor = modelo_global(avaliacoes)

    recomendacoes_np = recomendacoes_tensor.numpy()
    recomendacoes_df = pd.DataFrame(recomendacoes_np)

    ilv = IndividualLossVariance(avaliacoes_df, omega, 1)

    algorithmImpartiality_01_ma_np = AlgorithmImpartiality(avaliacoes_df, omega, 1)
    list_X_est = algorithmImpartiality_01_ma_np.evaluate(recomendacoes_df, 5) # calculates a list of h estimated matrices => h = 5

    list_losses = []
    for X_est in list_X_est:
        losses = ilv.get_losses(X_est)
        list_losses.append(losses)

    Z = AlgorithmImpartiality.losses_to_Z(list_losses, num_usuarios)
    list_Zs = AlgorithmImpartiality.matrices_Zs(Z, G)
    recomendacoes_fairness_np = AlgorithmImpartiality.make_matrix_X_gurobi(list_X_est, G, list_Zs) # recomendações com justiça
    return recomendacoes_fairness_np


def main():
    print("\n=== SERVIDOR (ETAPA DE TREINAMENTO INICIAL) ===")

    caminho_do_arquivo = 'X-u5-i10.xlsx'
    avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

    numero_de_usuarios = avaliacoes_inicial_tensor.shape[0]
    numero_de_itens = avaliacoes_inicial_tensor.shape[1]

    modelo_global_federado_01_ma_tensor = SimpleNN(numero_de_itens, 20, numero_de_itens)
    criterion = nn.MSELoss()

    # Realizando cópias do modelo global para cada uma das situações a serem analisadas
    treinar_modelo_global(modelo_global_federado_01_ma_tensor, avaliacoes_inicial_tensor, criterion)
    modelo_global_federado_02_mp_rindv_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_03_mp_loss_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_04_mp_nr_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_05_ma_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_06_mp_rindv_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_07_mp_loss_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_08_mp_nr_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_naofederado_09_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
   

    with torch.no_grad():
        recomendacoes_inicial_01_ma_tensor = modelo_global_federado_01_ma_tensor(avaliacoes_inicial_tensor)

    avaliacoes_final_01_ma_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_02_mp_rindv_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_03_mp_loss_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_04_mp_nr_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_05_ma_fairness_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_06_mp_rindv_fairness_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_07_mp_loss_fairness_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_08_mp_nr_fairness_tensor = copy.deepcopy(avaliacoes_inicial_tensor)
    avaliacoes_final_09_naofederado_tensor = copy.deepcopy(avaliacoes_inicial_tensor)

    G = {1: list(range(0, 2)), 2: list(range(2, 5))} # NR (Number Ratings)
    
    avaliacoes_random = [random.randint(1, 5) for _ in range(20)] # gerando uma lista de 20 avaliações aleatórias

    for round in range(2):
        print(f"\n=== ROUND {round} ===")

        print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
        print("treinar_modelos_locais :: modelos_clientes_01_ma_tensor")
        avaliacoes_final_01_ma_tensor, modelos_clientes_01_ma_tensor, calculos_modelos_clientes_01_ma, calculos_modelos_clientes_01_ma_loss_rindv, calculos_modelos_clientes_01_ma_nr_rindv = treinar_modelos_locais(modelo_global_federado_01_ma_tensor, avaliacoes_final_01_ma_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_02_mp_rindv_tensor")
        avaliacoes_final_02_mp_rindv_tensor, modelos_clientes_02_mp_rindv_tensor, calculos_modelos_clientes_02_mp_rindv_rindv, calculos_modelos_clientes_02_mp_rindv_loss, calculos_modelos_clientes_02_mp_rindv_nr = treinar_modelos_locais(modelo_global_federado_02_mp_rindv_tensor, avaliacoes_final_02_mp_rindv_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_03_mp_loss_tensor")
        avaliacoes_final_03_mp_loss_tensor, modelos_clientes_03_mp_loss_tensor, calculos_modelos_clientes_03_mp_loss_rindv, calculos_modelos_clientes_03_mp_loss_loss, calculos_modelos_clientes_03_mp_loss_nr = treinar_modelos_locais(modelo_global_federado_03_mp_loss_tensor, avaliacoes_final_03_mp_loss_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_04_mp_nr_tensor")
        avaliacoes_final_04_mp_nr_tensor, modelos_clientes_04_mp_nr_tensor, calculos_modelos_clientes_04_mp_nr_rindv, calculos_modelos_clientes_04_mp_nr_loss, calculos_modelos_clientes_04_mp_nr_nr = treinar_modelos_locais(modelo_global_federado_04_mp_nr_tensor, avaliacoes_final_04_mp_nr_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_05_ma_fairness_tensor")
        avaliacoes_final_05_ma_fairness_tensor, modelos_clientes_05_ma_fairness_tensor, calculos_modelos_clientes_05_ma_fairness_rindv, calculos_modelos_clientes_05_ma_loss_fairness_rindv, calculos_modelos_clientes_05_ma_nr_fairness_rindv = treinar_modelos_locais(modelo_global_federado_05_ma_fairness_tensor, avaliacoes_final_05_ma_fairness_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_06_mp_rindv_fairness_tensor")
        avaliacoes_final_06_mp_rindv_fairness_tensor, modelos_clientes_06_mp_rindv_fairness_tensor, calculos_modelos_clientes_06_mp_rindv_fairness_rindv, calculos_modelos_clientes_06_mp_loss_fairness_rindv, calculos_modelos_clientes_06_mp_nr_fairness_rindv = treinar_modelos_locais(modelo_global_federado_06_mp_rindv_fairness_tensor, avaliacoes_final_06_mp_rindv_fairness_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_07_mp_loss_fairness_tensor")
        avaliacoes_final_07_mp_loss_fairness_tensor, modelos_clientes_07_mp_loss_fairness_tensor, calculos_modelos_clientes_07_mp_rindv_fairness_loss, calculos_modelos_clientes_07_mp_loss_fairness_loss, calculos_modelos_clientes_07_mp_nr_fairness_loss = treinar_modelos_locais(modelo_global_federado_07_mp_loss_fairness_tensor, avaliacoes_final_07_mp_loss_fairness_tensor, avaliacoes_random, G, criterion)
        print("treinar_modelos_locais :: modelos_clientes_08_mp_nr_fairness_tensor")
        avaliacoes_final_08_mp_nr_fairness_tensor, modelos_clientes_08_mp_nr_fairness_tensor, calculos_modelos_clientes_08_mp_rindv_fairness_nr, calculos_modelos_clientes_08_mp_loss_fairness_nr, calculos_modelos_clientes_08_mp_nr_fairness_nr = treinar_modelos_locais(modelo_global_federado_08_mp_nr_fairness_tensor, avaliacoes_final_08_mp_nr_fairness_tensor, avaliacoes_random, G, criterion)

        print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL - AGREGAÇÃO) ===")
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado_01_ma_tensor")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado_01_ma_tensor, modelos_clientes_01_ma_tensor)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_rindv :: modelo_global_federado_02_mp_rindv_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global_federado_02_mp_rindv_tensor, modelos_clientes_02_mp_rindv_tensor, calculos_modelos_clientes_02_mp_rindv_rindv)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_loss :: modelo_global_federado_03_mp_loss_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global_federado_03_mp_loss_tensor, modelos_clientes_03_mp_loss_tensor, calculos_modelos_clientes_03_mp_loss_loss)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_nr :: modelo_global_federado_04_mp_nr_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global_federado_04_mp_nr_tensor, modelos_clientes_04_mp_nr_tensor, calculos_modelos_clientes_04_mp_nr_nr)
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado_05_ma_fairness_tensor")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado_05_ma_fairness_tensor, modelos_clientes_05_ma_fairness_tensor)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_rindv :: modelo_global_federado_06_mp_rindv_fairness_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global_federado_06_mp_rindv_fairness_tensor, modelos_clientes_06_mp_rindv_fairness_tensor, calculos_modelos_clientes_06_mp_rindv_fairness_rindv)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_loss :: modelo_global_federado_07_mp_loss_fairness_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global_federado_07_mp_loss_fairness_tensor, modelos_clientes_07_mp_loss_fairness_tensor, calculos_modelos_clientes_07_mp_loss_fairness_loss)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_nr :: modelo_global_federado_08_mp_nr_fairness_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global_federado_08_mp_nr_fairness_tensor, modelos_clientes_08_mp_nr_fairness_tensor, calculos_modelos_clientes_08_mp_nr_fairness_nr)

        # Agrupamento dos usuários no Sistema de Recomendação Federado com Justiça
        # ??? Verificar se posso treinar o modelo baseado nas recomendações e não somente nas avaliações (como feito para os outros). Ou, ver se consigo gerar um modelo à partir dos tensores (mas de outra forma)
        recomendacoes_final_05_ma_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_05_ma_fairness_tensor, avaliacoes_final_05_ma_fairness_tensor, G)
        recomendacoes_final_05_ma_fairness_tensor = torch.tensor(recomendacoes_final_05_ma_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_05_ma_fairness_tensor, recomendacoes_final_05_ma_fairness_tensor, criterion)
        
        recomendacoes_final_06_mp_rindv_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_06_mp_rindv_fairness_tensor, avaliacoes_final_06_mp_rindv_fairness_tensor, G)
        recomendacoes_final_06_mp_rindv_fairness_tensor = torch.tensor(recomendacoes_final_06_mp_rindv_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_06_mp_rindv_fairness_tensor, recomendacoes_final_06_mp_rindv_fairness_tensor, criterion)
        
        recomendacoes_final_07_mp_loss_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_07_mp_loss_fairness_tensor, avaliacoes_final_07_mp_loss_fairness_tensor, G)
        recomendacoes_final_07_mp_loss_fairness_tensor = torch.tensor(recomendacoes_final_07_mp_loss_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_07_mp_loss_fairness_tensor, recomendacoes_final_07_mp_loss_fairness_tensor, criterion)

        recomendacoes_final_08_mp_nr_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_08_mp_nr_fairness_tensor, avaliacoes_final_08_mp_nr_fairness_tensor, G)
        recomendacoes_final_08_mp_nr_fairness_tensor = torch.tensor(recomendacoes_final_08_mp_nr_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_08_mp_nr_fairness_tensor, recomendacoes_final_08_mp_nr_fairness_tensor, criterion)
        
        # Agrupamento dos usuários no Sistema de Recomendação NÃO Federado
        avaliacoes_final_09_naofederado_tensor = copy.deepcopy(avaliacoes_final_01_ma_tensor)
        treinar_modelo_global(modelo_global_naofederado_09_tensor, avaliacoes_final_09_naofederado_tensor, criterion)

    with torch.no_grad():
        recomendacoes_final_01_ma_tensor = modelo_global_federado_01_ma_tensor(avaliacoes_final_01_ma_tensor)
        recomendacoes_final_02_mp_rindv_tensor = modelo_global_federado_02_mp_rindv_tensor(avaliacoes_final_02_mp_rindv_tensor)
        recomendacoes_final_03_mp_loss_tensor = modelo_global_federado_03_mp_loss_tensor(avaliacoes_final_03_mp_loss_tensor)
        recomendacoes_final_04_mp_nr_tensor = modelo_global_federado_04_mp_nr_tensor(avaliacoes_final_04_mp_nr_tensor)
        recomendacoes_final_05_ma_fairness_tensor = modelo_global_federado_05_ma_fairness_tensor(avaliacoes_final_05_ma_fairness_tensor)
        recomendacoes_final_06_mp_rindv_fairness_tensor = modelo_global_federado_06_mp_rindv_fairness_tensor(avaliacoes_final_06_mp_rindv_fairness_tensor)
        recomendacoes_final_07_mp_loss_fairness_tensor = modelo_global_federado_07_mp_loss_fairness_tensor(avaliacoes_final_07_mp_loss_fairness_tensor)
        recomendacoes_final_08_mp_nr_fairness_tensor = modelo_global_federado_08_mp_nr_fairness_tensor(avaliacoes_final_08_mp_nr_fairness_tensor)
        recomendacoes_final_09_naofederado_tensor = modelo_global_naofederado_09_tensor(avaliacoes_final_09_naofederado_tensor)
    
    print("\n=== MEDIDA DE JUSTIÇA ===")
    avaliacoes_inicial_np = avaliacoes_inicial_tensor.numpy()
    avaliacoes_inicial_df = pd.DataFrame(avaliacoes_inicial_np)
    avaliacoes_final_01_ma_np = avaliacoes_final_01_ma_tensor.numpy()
    avaliacoes_final_01_ma_df = pd.DataFrame(avaliacoes_final_01_ma_np)
    avaliacoes_final_02_mp_rindv_np = avaliacoes_final_02_mp_rindv_tensor.numpy()
    avaliacoes_final_02_mp_rindv_df = pd.DataFrame(avaliacoes_final_02_mp_rindv_np)
    avaliacoes_final_03_mp_loss_np = avaliacoes_final_03_mp_loss_tensor.numpy()
    avaliacoes_final_03_mp_loss_df = pd.DataFrame(avaliacoes_final_03_mp_loss_np)
    avaliacoes_final_04_mp_nr_np = avaliacoes_final_04_mp_nr_tensor.numpy()
    avaliacoes_final_04_mp_nr_df = pd.DataFrame(avaliacoes_final_04_mp_nr_np)
    avaliacoes_final_05_ma_fairness_np = avaliacoes_final_05_ma_fairness_tensor.numpy()
    avaliacoes_final_05_ma_fairness_df = pd.DataFrame(avaliacoes_final_05_ma_fairness_np)
    avaliacoes_final_06_mp_rindv_fairness_np = avaliacoes_final_06_mp_rindv_fairness_tensor.numpy()
    avaliacoes_final_06_mp_rindv_fairness_df = pd.DataFrame(avaliacoes_final_06_mp_rindv_fairness_np)
    avaliacoes_final_07_mp_loss_fairness_np = avaliacoes_final_07_mp_loss_fairness_tensor.numpy()
    avaliacoes_final_07_mp_loss_fairness_df = pd.DataFrame(avaliacoes_final_07_mp_loss_fairness_np)
    avaliacoes_final_08_mp_nr_fairness_np = avaliacoes_final_08_mp_nr_fairness_tensor.numpy()
    avaliacoes_final_08_mp_nr_fairness_df = pd.DataFrame(avaliacoes_final_08_mp_nr_fairness_np)
    avaliacoes_final_09_naofederado_np = avaliacoes_final_09_naofederado_tensor.numpy()
    avaliacoes_final_09_naofederado_df = pd.DataFrame(avaliacoes_final_09_naofederado_np)

    recomendacoes_inicial_01_ma_np = recomendacoes_inicial_01_ma_tensor.numpy()
    recomendacoes_inicial_01_ma_df = pd.DataFrame(recomendacoes_inicial_01_ma_np)
    recomendacoes_final_01_ma_np = recomendacoes_final_01_ma_tensor.numpy()
    recomendacoes_final_01_ma_df = pd.DataFrame(recomendacoes_final_01_ma_np)
    recomendacoes_final_02_mp_rindv_np = recomendacoes_final_02_mp_rindv_tensor.numpy()
    recomendacoes_final_02_mp_rindv_df = pd.DataFrame(recomendacoes_final_02_mp_rindv_np)
    recomendacoes_final_03_mp_loss_np = recomendacoes_final_03_mp_loss_tensor.numpy()
    recomendacoes_final_03_mp_loss_df = pd.DataFrame(recomendacoes_final_03_mp_loss_np)
    recomendacoes_final_04_mp_nr_np = recomendacoes_final_04_mp_nr_tensor.numpy()
    recomendacoes_final_04_mp_nr_df = pd.DataFrame(recomendacoes_final_04_mp_nr_np)
    recomendacoes_final_05_ma_fairness_np = recomendacoes_final_05_ma_fairness_tensor.numpy()
    recomendacoes_final_05_ma_fairness_df = pd.DataFrame(recomendacoes_final_05_ma_fairness_np)
    recomendacoes_final_06_mp_rindv_fairness_np = recomendacoes_final_06_mp_rindv_fairness_tensor.numpy()
    recomendacoes_final_06_mp_rindv_fairness_df = pd.DataFrame(recomendacoes_final_06_mp_rindv_fairness_np)
    recomendacoes_final_07_mp_loss_fairness_np = recomendacoes_final_07_mp_loss_fairness_tensor.numpy()
    recomendacoes_final_07_mp_loss_fairness_df = pd.DataFrame(recomendacoes_final_07_mp_loss_fairness_np)
    recomendacoes_final_08_mp_nr_fairness_np = recomendacoes_final_08_mp_nr_fairness_tensor.numpy()
    recomendacoes_final_08_mp_nr_fairness_df = pd.DataFrame(recomendacoes_final_08_mp_nr_fairness_np)
    recomendacoes_final_09_naofederado_np = recomendacoes_final_09_naofederado_tensor.numpy()
    recomendacoes_final_09_naofederado_df = pd.DataFrame(recomendacoes_final_09_naofederado_np)

    omega_inicial = (avaliacoes_inicial_df != 0)
    omega_final_01_ma = (avaliacoes_final_01_ma_df != 0)    
    omega_final_02_mp_rindv = (avaliacoes_final_02_mp_rindv_df != 0)    
    omega_final_03_mp_loss = (avaliacoes_final_03_mp_loss_df != 0)    
    omega_final_04_mp_nr = (avaliacoes_final_04_mp_nr_df != 0)    
    omega_final_05_ma_fairness = (avaliacoes_final_05_ma_fairness_df != 0)
    omega_final_06_mp_rindv_fairness = (avaliacoes_final_06_mp_rindv_fairness_df != 0)
    omega_final_07_mp_loss_fairness = (avaliacoes_final_07_mp_loss_fairness_df != 0)
    omega_final_08_mp_nr_fairness = (avaliacoes_final_08_mp_nr_fairness_df != 0)
    omega_final_09_naofederado = (avaliacoes_final_09_naofederado_df != 0)

    # To capture polarization, we seek to measure the extent to which the user ratings disagree
    polarization = Polarization()
    Rpol_inicial = polarization.evaluate(recomendacoes_inicial_01_ma_df)
    Rpol_final_01_ma = polarization.evaluate(recomendacoes_final_01_ma_df)
    Rpol_final_02_mp_rindv = polarization.evaluate(recomendacoes_final_02_mp_rindv_df)
    Rpol_final_03_mp_loss = polarization.evaluate(recomendacoes_final_03_mp_loss_df)
    Rpol_final_04_mp_nr = polarization.evaluate(recomendacoes_final_04_mp_nr_df)
    Rpol_final_05_ma_fairness = polarization.evaluate(recomendacoes_final_05_ma_fairness_df)
    Rpol_final_06_mp_rindv_fairness = polarization.evaluate(recomendacoes_final_06_mp_rindv_fairness_df)
    Rpol_final_07_mp_loss_fairness = polarization.evaluate(recomendacoes_final_07_mp_loss_fairness_df)
    Rpol_final_08_mp_nr_fairness = polarization.evaluate(recomendacoes_final_08_mp_nr_fairness_df)
    Rpol_final_09_naofederado = polarization.evaluate(recomendacoes_final_09_naofederado_df)
    print(f"\nPolarization Inicial (Rpol)                                        : {Rpol_inicial:.9f}")
    print(f"Polarization Final   (Rpol [1 :: Média Aritmética               ]) : {Rpol_final_01_ma:.9f}")
    print(f"Polarization Final   (Rpol [2 :: Média Ponderada Rindv          ]) : {Rpol_final_02_mp_rindv:.9f}")
    print(f"Polarization Final   (Rpol [3 :: Média Ponderada Loss           ]) : {Rpol_final_03_mp_loss:.9f}")
    print(f"Polarization Final   (Rpol [4 :: Média Ponderada NR             ]) : {Rpol_final_04_mp_nr:.9f}")
    print(f"Polarization Final   (Rpol [5 :: Média Aritmética Fairness      ]) : {Rpol_final_05_ma_fairness:.9f}")
    print(f"Polarization Final   (Rpol [6 :: Média Ponderada Fairness Rindv ]) : {Rpol_final_06_mp_rindv_fairness:.9f}")
    print(f"Polarization Final   (Rpol [7 :: Média Ponderada Fairness Loss  ]) : {Rpol_final_07_mp_loss_fairness:.9f}")
    print(f"Polarization Final   (Rpol [8 :: Média Ponderada Fairness NR    ]) : {Rpol_final_08_mp_nr_fairness:.9f}")
    print(f"Polarization Final   (Rpol [9 :: Não Federado                   ]) : {Rpol_final_09_naofederado:.9f}")

    ilv_inicial = IndividualLossVariance(avaliacoes_inicial_df, omega_inicial, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_01_ma = IndividualLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_02_mp_rindv = IndividualLossVariance(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_03_mp_loss = IndividualLossVariance(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_04_mp_nr = IndividualLossVariance(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_05_ma_fairness = IndividualLossVariance(avaliacoes_final_05_ma_fairness_df, omega_final_05_ma_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_06_mp_rindv_fairness = IndividualLossVariance(avaliacoes_final_06_mp_rindv_fairness_df, omega_final_06_mp_rindv_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_07_mp_loss_fairness = IndividualLossVariance(avaliacoes_final_07_mp_loss_fairness_df, omega_final_07_mp_loss_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_08_mp_nr_fairness = IndividualLossVariance(avaliacoes_final_08_mp_nr_fairness_df, omega_final_08_mp_nr_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_09_naofederado = IndividualLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, 1) #axis = 1 (0 rows e 1 columns)

    Rindv_inicial = ilv_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    Rindv_final_01_ma = ilv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    Rindv_final_02_mp_rindv = ilv_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    Rindv_final_03_mp_loss = ilv_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    Rindv_final_04_mp_nr = ilv_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    Rindv_final_05_ma_fairness = ilv_final_05_ma_fairness.evaluate(recomendacoes_final_05_ma_fairness_df)
    Rindv_final_06_mp_rindv_fairness = ilv_final_06_mp_rindv_fairness.evaluate(recomendacoes_final_06_mp_rindv_fairness_df)
    Rindv_final_07_mp_loss_fairness = ilv_final_07_mp_loss_fairness.evaluate(recomendacoes_final_07_mp_loss_fairness_df)
    Rindv_final_08_mp_nr_fairness = ilv_final_08_mp_nr_fairness.evaluate(recomendacoes_final_08_mp_nr_fairness_df)
    Rindv_final_09_naofederado = ilv_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)
    print(f"\nIndividual Loss Variance (Rindv Inicial)                                      : {Rindv_inicial:.9f}")
    print(f"Individual Loss Variance (Rindv Final [1 :: Média Aritmética               ]) : {Rindv_final_01_ma:.9f}")
    print(f"Individual Loss Variance (Rindv Final [2 :: Média Ponderada Rindv          ]) : {Rindv_final_02_mp_rindv:.9f}")
    print(f"Individual Loss Variance (Rindv Final [3 :: Média Ponderada Loss           ]) : {Rindv_final_03_mp_loss:.9f}")
    print(f"Individual Loss Variance (Rindv Final [4 :: Média Ponderada NR             ]) : {Rindv_final_04_mp_nr:.9f}")
    print(f"Individual Loss Variance (Rindv Final [5 :: Média Aritmética       Fairness]) : {Rindv_final_05_ma_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [6 :: Média Ponderada Rindv Fairness ]) : {Rindv_final_06_mp_rindv_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [7 :: Média Ponderada Loss Fairness  ]) : {Rindv_final_07_mp_loss_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [8 :: Média Ponderada NR Fairness    ]) : {Rindv_final_08_mp_nr_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [9 :: Não Federado                   ]) : {Rindv_final_09_naofederado:.9f}")

    print("G")
    print(G[1])

    glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_01_ma = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_02_mp_rindv = GroupLossVariance(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_03_mp_loss = GroupLossVariance(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_04_mp_nr = GroupLossVariance(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_05_ma_fairness_rindv = GroupLossVariance(avaliacoes_final_05_ma_fairness_df, omega_final_05_ma_fairness, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_06_mp_fairness_rindv = GroupLossVariance(avaliacoes_final_06_mp_rindv_fairness_df, omega_final_06_mp_rindv_fairness, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_07_mp_fairness_loss = GroupLossVariance(avaliacoes_final_07_mp_loss_fairness_df, omega_final_07_mp_loss_fairness, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_08_mp_fairness_nr = GroupLossVariance(avaliacoes_final_08_mp_nr_fairness_df, omega_final_08_mp_nr_fairness, G, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_09_naofederado = GroupLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, G, 1) #axis = 1 (0 rows e 1 columns)

    RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    RgrpNR_final_01_ma = glv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    RgrpNR_final_02_mp_rindv = glv_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    RgrpNR_final_03_mp_loss = glv_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    RgrpNR_final_04_mp_nr = glv_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    RgrpNR_final_05_ma_fairness_rindv = glv_final_05_ma_fairness_rindv.evaluate(recomendacoes_final_05_ma_fairness_df)
    RgrpNR_final_06_mp_fairness_rindv = glv_final_06_mp_fairness_rindv.evaluate(recomendacoes_final_06_mp_rindv_fairness_df)
    RgrpNR_final_07_mp_fairness_loss = glv_final_07_mp_fairness_loss.evaluate(recomendacoes_final_07_mp_loss_fairness_df)
    RgrpNR_final_08_mp_fairness_nr = glv_final_08_mp_fairness_nr.evaluate(recomendacoes_final_08_mp_nr_fairness_df)
    RgrpNR_final_09_naofederado = glv_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)

    print(f"\nGroup Loss Variance (Rgrp Inicial)                                      : {RgrpNR_inicial:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética               ]) : {RgrpNR_final_01_ma:.9f}")
    print(f"Group Loss Variance (Rgrp Final [2 :: Média Ponderada Rindv          ]) : {RgrpNR_final_02_mp_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [3 :: Média Ponderada Loss           ]) : {RgrpNR_final_03_mp_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [4 :: Média Ponderada NR             ]) : {RgrpNR_final_04_mp_nr:.9f}")
    print(f"Group Loss Variance (Rgrp Final [5 :: Média Aritmética Fairness      ]) : {RgrpNR_final_05_ma_fairness_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [6 :: Média Ponderada Rindv Fairness ]) : {RgrpNR_final_06_mp_fairness_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [7 :: Média Ponderada Loss Fairness  ]) : {RgrpNR_final_07_mp_fairness_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [8 :: Média Ponderada NR Fairness    ]) : {RgrpNR_final_08_mp_fairness_nr:.9f}")
    print(f"Group Loss Variance (Rgrp Final [9 :: Não Federado                   ]) : {RgrpNR_final_09_naofederado:.9f}")

    rmse_inicial = RMSE(avaliacoes_inicial_df, omega_inicial)
    rmse_final_01_ma = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)
    rmse_final_02_mp_rindv = RMSE(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv)
    rmse_final_03_mp_loss = RMSE(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss)
    rmse_final_04_mp_nr = RMSE(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr)
    rmse_final_05_ma_fairness_rindv = RMSE(avaliacoes_final_05_ma_fairness_df, omega_final_05_ma_fairness)
    rmse_final_06_mp_fairness_rindv = RMSE(avaliacoes_final_06_mp_rindv_fairness_df, omega_final_06_mp_rindv_fairness)
    rmse_final_07_mp_fairness_loss = RMSE(avaliacoes_final_07_mp_loss_fairness_df, omega_final_07_mp_loss_fairness)
    rmse_final_08_mp_fairness_nr = RMSE(avaliacoes_final_08_mp_nr_fairness_df, omega_final_08_mp_nr_fairness)
    rmse_final_09_naofederado = RMSE(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado)

    result_inicial = rmse_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    result_final_01_ma = rmse_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    result_final_02_mp_rindv = rmse_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    result_final_03_mp_loss = rmse_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    result_final_04_mp_nr = rmse_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    result_final_05_ma_fairness_rindv = rmse_final_05_ma_fairness_rindv.evaluate(recomendacoes_final_05_ma_fairness_df)
    result_final_06_mp_fairness_rindv = rmse_final_06_mp_fairness_rindv.evaluate(recomendacoes_final_06_mp_rindv_fairness_df)
    result_final_07_mp_fairness_loss = rmse_final_07_mp_fairness_loss.evaluate(recomendacoes_final_07_mp_loss_fairness_df)
    result_final_08_mp_fairness_nr = rmse_final_08_mp_fairness_nr.evaluate(recomendacoes_final_08_mp_nr_fairness_df)
    result_final_09_naofederado = rmse_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)

    print(f'\nRMSE Inicial                                      : {result_inicial:.9f}')
    print(f'RMSE Final [1 :: Média Aritmética               ] : {result_final_01_ma:.9f}')
    print(f'RMSE Final [2 :: Média Ponderada Rindv          ] : {result_final_02_mp_rindv:.9f}')
    print(f'RMSE Final [3 :: Média Ponderada Loss           ] : {result_final_03_mp_loss:.9f}')
    print(f'RMSE Final [4 :: Média Ponderada NR             ] : {result_final_04_mp_nr:.9f}')
    print(f'RMSE Final [5 :: Média Aritmética Fairness      ] : {result_final_05_ma_fairness_rindv:.9f}')
    print(f'RMSE Final [6 :: Média Ponderada Fairness Rindv ] : {result_final_06_mp_fairness_rindv:.9f}')
    print(f'RMSE Final [7 :: Média Ponderada Fairness Loss  ] : {result_final_07_mp_fairness_loss:.9f}')
    print(f'RMSE Final [8 :: Média Ponderada Fairness NR    ] : {result_final_08_mp_fairness_nr:.9f}')
    print(f'RMSE Final [9 :: Não Federado                   ] : {result_final_09_naofederado:.9f}')

    # print("--------------------------------------------------------------------------------")
    # G_Gender = {1: [183, 200, 272, 247, 220, 212, 251, 141, 175, 137, 99, 299, 94, 230, 87, 86, 7, 270, 60, 120, 48, 170, 205, 16, 30, 160, 41, 37, 96, 261, 69, 129, 176, 107, 19, 266, 123, 32, 13, 84, 271, 134, 64, 124, 288, 1, 227, 257, 279, 159, 285, 74, 276, 18, 248, 135, 67, 252, 80, 117, 194, 26, 17, 127, 56, 61, 138, 102, 295, 76, 126, 3, 77, 296, 207, 177, 292, 156, 161, 108, 27, 165, 191, 51, 4, 174, 5, 269, 168, 283, 70, 21, 209, 162, 289, 232, 198, 82, 66, 249, 273, 54, 238, 281, 72, 297, 184, 293, 105, 149, 49, 203, 188, 258, 10, 196, 210, 71, 139, 291, 226, 33, 233, 52, 142, 186, 118, 166, 189, 23, 95, 112, 50, 151, 181, 45, 88, 12, 89, 231, 55, 68, 147, 294, 169, 125, 208, 103, 93, 85, 28, 286, 259, 39, 58, 222, 267, 211, 223, 201, 152, 245, 224, 178, 143, 140, 229, 47, 154, 110, 277, 136, 0, 206, 40, 100, 250, 2, 65, 11, 278, 29, 246, 36, 83, 15, 22, 20, 24, 234, 287, 146, 256, 130, 280, 38, 131, 187, 218, 57, 164, 128, 78, 282, 172, 43, 44, 182, 132, 115, 298, 219, 75, 109, 265, 8, 90, 260, 239, 148, 202, 153, 262, 53, 79, 59, 263, 255, 284, 204, 9, 237, 6, 111, 268, 240, 274, 46, 275, 62], 2: [180, 221, 92, 145, 199, 34, 73, 14, 35, 213, 133, 225, 81, 63, 101, 155, 113, 214, 97, 195, 236, 215, 171, 91, 121, 185, 98, 116, 31, 114, 216, 157, 217, 167, 254, 42, 190, 197, 241, 243, 228, 104, 242, 264, 163, 144, 244, 235, 179, 122, 173, 150, 290, 158, 25, 253, 192, 119, 193, 106]}
    
    # print("G_Gender")
    # print(G_Gender[1])

    # glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_01_ma = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_02_mp_rindv = GroupLossVariance(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_03_mp_loss = GroupLossVariance(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_04_mp_nr = GroupLossVariance(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_05_ma_fairness_rindv = GroupLossVariance(avaliacoes_final_05_ma_fairness_df, omega_final_05_ma_fairness, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_06_mp_fairness_rindv = GroupLossVariance(avaliacoes_final_06_mp_rindv_fairness_df, omega_final_06_mp_rindv_fairness, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_07_mp_fairness_loss = GroupLossVariance(avaliacoes_final_07_mp_loss_fairness_df, omega_final_07_mp_loss_fairness, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_08_mp_fairness_nr = GroupLossVariance(avaliacoes_final_08_mp_nr_fairness_df, omega_final_08_mp_nr_fairness, G_Gender, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_09_naofederado = GroupLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, G_Gender, 1) #axis = 1 (0 rows e 1 columns)

    # RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    # RgrpNR_final_01_ma = glv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    # RgrpNR_final_02_mp_rindv = glv_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    # RgrpNR_final_03_mp_loss = glv_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    # RgrpNR_final_04_mp_nr = glv_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    # RgrpNR_final_05_ma_fairness_rindv = glv_final_05_ma_fairness_rindv.evaluate(recomendacoes_final_05_ma_fairness_df)
    # RgrpNR_final_06_mp_fairness_rindv = glv_final_06_mp_fairness_rindv.evaluate(recomendacoes_final_06_mp_rindv_fairness_df)
    # RgrpNR_final_07_mp_fairness_loss = glv_final_07_mp_fairness_loss.evaluate(recomendacoes_final_07_mp_loss_fairness_df)
    # RgrpNR_final_08_mp_fairness_nr = glv_final_08_mp_fairness_nr.evaluate(recomendacoes_final_08_mp_nr_fairness_df)
    # RgrpNR_final_09_naofederado = glv_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)

    # print(f"\nGroup Loss Variance (Rgrp Inicial)                                      : {RgrpNR_inicial:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética               ]) : {RgrpNR_final_01_ma:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [2 :: Média Ponderada Rindv          ]) : {RgrpNR_final_02_mp_rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [3 :: Média Ponderada Loss           ]) : {RgrpNR_final_03_mp_loss:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [4 :: Média Ponderada NR             ]) : {RgrpNR_final_04_mp_nr:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [5 :: Média Aritmética Fairness      ]) : {RgrpNR_final_05_ma_fairness_rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [6 :: Média Ponderada Rindv Fairness ]) : {RgrpNR_final_06_mp_fairness_rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [7 :: Média Ponderada Loss Fairness  ]) : {RgrpNR_final_07_mp_fairness_loss:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [8 :: Média Ponderada NR Fairness    ]) : {RgrpNR_final_08_mp_fairness_nr:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [9 :: Não Federado                   ]) : {RgrpNR_final_09_naofederado:.9f}")

    # G_Age = {1: [14, 194, 273, 132, 262], 2: [251, 175, 94, 86, 270, 48, 92, 96, 129, 107, 134, 64, 124, 288, 159, 26, 101, 61, 76, 126, 207, 191, 174, 168, 70, 215, 209, 171, 82, 149, 71, 33, 216, 157, 189, 23, 50, 231, 222, 201, 140, 163, 246, 244, 282, 265, 8, 90, 290, 158, 255, 237, 275], 3: [183, 200, 220, 212, 141, 230, 7, 60, 120, 170, 205, 16, 41, 145, 37, 34, 261, 69, 176, 32, 73, 35, 213, 133, 279, 285, 74, 276, 135, 252, 80, 225, 81, 63, 56, 138, 102, 295, 3, 296, 155, 161, 108, 51, 97, 269, 195, 283, 236, 21, 289, 232, 198, 66, 249, 238, 72, 293, 203, 188, 258, 10, 196, 139, 116, 291, 226, 233, 31, 142, 186, 118, 217, 151, 181, 45, 254, 42, 190, 89, 55, 147, 294, 169, 103, 93, 85, 28, 286, 39, 241, 267, 211, 223, 143, 229, 104, 110, 277, 136, 206, 40, 65, 11, 264, 29, 15, 22, 24, 234, 287, 130, 280, 131, 187, 164, 128, 43, 44, 298, 219, 75, 179, 109, 122, 173, 260, 202, 53, 79, 59, 253, 192, 204, 9, 119, 6, 111, 268, 240, 193, 106], 4: [272, 99, 299, 87, 199, 13, 84, 271, 1, 227, 18, 248, 117, 17, 127, 77, 177, 292, 156, 27, 214, 4, 5, 91, 297, 105, 49, 121, 98, 210, 52, 166, 95, 112, 88, 68, 208, 243, 152, 245, 228, 100, 250, 2, 278, 36, 144, 146, 256, 38, 57, 78, 172, 182, 150, 153, 25, 263], 5: [137, 221, 30, 160, 19, 67, 165, 113, 54, 281, 184, 167, 197, 125, 58, 47, 242, 83, 20, 235, 239, 148, 46, 62], 6: [247, 180, 266, 123, 257, 185, 114, 224, 178, 0, 115, 274], 7: [162, 12, 259, 154, 218, 284]}
    
    # print("G_Age")
    # print(G_Age[1])

    # glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_01_ma = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_02_mp_rindv = GroupLossVariance(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_03_mp_loss = GroupLossVariance(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_04_mp_nr = GroupLossVariance(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_05_ma_fairness_rindv = GroupLossVariance(avaliacoes_final_05_ma_fairness_df, omega_final_05_ma_fairness, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_06_mp_fairness_rindv = GroupLossVariance(avaliacoes_final_06_mp_rindv_fairness_df, omega_final_06_mp_rindv_fairness, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_07_mp_fairness_loss = GroupLossVariance(avaliacoes_final_07_mp_loss_fairness_df, omega_final_07_mp_loss_fairness, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_08_mp_fairness_nr = GroupLossVariance(avaliacoes_final_08_mp_nr_fairness_df, omega_final_08_mp_nr_fairness, G_Age, 1) #axis = 1 (0 rows e 1 columns)
    # glv_final_09_naofederado = GroupLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, G_Age, 1) #axis = 1 (0 rows e 1 columns)

    # RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    # RgrpNR_final_01_ma = glv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    # RgrpNR_final_02_mp_rindv = glv_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    # RgrpNR_final_03_mp_loss = glv_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    # RgrpNR_final_04_mp_nr = glv_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    # RgrpNR_final_05_ma_fairness_rindv = glv_final_05_ma_fairness_rindv.evaluate(recomendacoes_final_05_ma_fairness_df)
    # RgrpNR_final_06_mp_fairness_rindv = glv_final_06_mp_fairness_rindv.evaluate(recomendacoes_final_06_mp_rindv_fairness_df)
    # RgrpNR_final_07_mp_fairness_loss = glv_final_07_mp_fairness_loss.evaluate(recomendacoes_final_07_mp_loss_fairness_df)
    # RgrpNR_final_08_mp_fairness_nr = glv_final_08_mp_fairness_nr.evaluate(recomendacoes_final_08_mp_nr_fairness_df)
    # RgrpNR_final_09_naofederado = glv_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)

    # print(f"\nGroup Loss Variance (Rgrp Inicial)                                      : {RgrpNR_inicial:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética               ]) : {RgrpNR_final_01_ma:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [2 :: Média Ponderada Rindv          ]) : {RgrpNR_final_02_mp_rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [3 :: Média Ponderada Loss           ]) : {RgrpNR_final_03_mp_loss:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [4 :: Média Ponderada NR             ]) : {RgrpNR_final_04_mp_nr:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [5 :: Média Aritmética Fairness      ]) : {RgrpNR_final_05_ma_fairness_rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [6 :: Média Ponderada Rindv Fairness ]) : {RgrpNR_final_06_mp_fairness_rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [7 :: Média Ponderada Loss Fairness  ]) : {RgrpNR_final_07_mp_fairness_loss:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [8 :: Média Ponderada NR Fairness    ]) : {RgrpNR_final_08_mp_fairness_nr:.9f}")
    # print(f"Group Loss Variance (Rgrp Final [9 :: Não Federado                   ]) : {RgrpNR_final_09_naofederado:.9f}")

    # avaliacoes_inicial_df.to_excel("avaliacoes_inicial.xlsx", index=False)
    # avaliacoes_final_df.to_excel("avaliacoes_final.xlsx", index=False)
    # recomendacoes_inicial_df.to_excel("recomendacoes_inicial.xlsx", index=False)
    # recomendacoes_final_df1.to_excel("recomendacoes_final_df1.xlsx", index=False)
    # recomendacoes_final_df2.to_excel("recomendacoes_final_df2.xlsx", index=False)
    # recomendacoes_final_df3.to_excel("recomendacoes_final_df3.xlsx", index=False)
    # recomendacoes_final_df4.to_excel("recomendacoes_final_df4.xlsx", index=False)
    # recomendacoes_modelo_global_nao_federado_df.to_excel("recomendacoes_modelo_global_nao_federado_df.xlsx", index=False)

    print("\navaliacoes_inicial_df")
    print(avaliacoes_inicial_df)

    print("\nrecomendacoes_inicial_01_ma_df")
    print(recomendacoes_inicial_01_ma_df)

    print("\navaliacoes_final_09_naofederado_df")
    print(avaliacoes_final_09_naofederado_df)

    print("\nrecomendacoes_final_09_naofederado_df")
    print(recomendacoes_final_09_naofederado_df)

    avaliacoes_inicial_df.to_excel("X-u10-avaliacoes_inicial.xlsx", index=False)
    recomendacoes_inicial_01_ma_df.to_excel("X-u10-recomendacoes_inicial_01_ma_df.xlsx", index=False)
    avaliacoes_final_09_naofederado_df.to_excel("X-u10-avaliacoes_final_09_naofederado_df.xlsx", index=False)
    recomendacoes_final_09_naofederado_df.to_excel("X-u10-recomendacoes_final_09_naofederado.xlsx", index=False)

if __name__ == "__main__":
    main()
