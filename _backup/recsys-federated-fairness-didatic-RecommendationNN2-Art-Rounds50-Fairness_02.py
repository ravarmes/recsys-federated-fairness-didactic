import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance
from AlgorithmImpartiality import AlgorithmImpartiality

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

def carregar_avaliacoes_do_arquivo_txt(caminho_do_arquivo):
    dados = np.loadtxt(caminho_do_arquivo, delimiter=',', dtype=np.float32)
    return torch.tensor(dados), dados
    
def treinar_modelo_global(modelo_global, avaliacoes, criterion, epochs=1, learning_rate=0.5):
    optimizer = optim.SGD(modelo_global.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(modelo_global.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    num_usuarios, num_itens = avaliacoes.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten(), itens_ids.flatten()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = modelo_global(usuarios_ids.long(), itens_ids.long()).view(num_usuarios, num_itens)
        loss = criterion(predictions, avaliacoes.float())
        loss.backward()
        optimizer.step()

def treinar_modelos_locais(modelo_global, avaliacoes, criterion, epochs=1, learning_rate=0.5):
    # Inicialização de dados e listas
    avaliacoes_final = avaliacoes.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes.size(0))] # criando uma cópia de modelo global inicial para cada usuário
    modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr = [], [], []
    
    NUMBER_ADVANTAGED_GROUP = 15
    NR_ADVANTAGED_GROUP = 5
    NR_DISADVANTAGED_GROUP = 1

    num_usuarios, num_itens = avaliacoes.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten().long(), itens_ids.flatten().long()

    for i, modelo_cliente in enumerate(modelos_clientes):
        # print(f"=== Treinamento no Cliente {i + 1} ===")
        indices_nao_avaliados = (avaliacoes[i] == 0).nonzero(as_tuple=False).squeeze()

        indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP]]
        novas_avaliacoes = torch.randint(1, 6, (NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP,)).float()
        modelos_clientes_nr.append((i, NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP))

        avaliacoes_final[i, indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final_cliente = avaliacoes.clone()
        avaliacoes_final_cliente[i, indices_novas_avaliacoes] = novas_avaliacoes

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer_cliente = optim.Adam(modelo_cliente.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        for epoch in range(epochs):
            optimizer_cliente.zero_grad()
            predictions = modelo_cliente(usuarios_ids, itens_ids).view(num_usuarios, num_itens).float()
            loss_cliente = criterion(predictions, avaliacoes_final_cliente)
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(usuarios_ids, itens_ids).view(num_usuarios, num_itens)

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

    return avaliacoes_final, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr

def treinar_modelos_locais2(modelo_global, avaliacoes, criterion, epochs=1, learning_rate=0.5, G = {1:list(range(0, 15)), 2:list(range(15, 300))}):
    # Inicialização de dados e listas
    avaliacoes_final = avaliacoes.clone()
    modelos_clientes = [copy.deepcopy(modelo_global) for _ in range(avaliacoes.size(0))] # criando uma cópia de modelo global inicial para cada usuário
    modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr = [], [], []
    
    NUMBER_ADVANTAGED_GROUP = 15
    NR_ADVANTAGED_GROUP = 5
    NR_DISADVANTAGED_GROUP = 1

    num_usuarios, num_itens = avaliacoes.shape
    usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
    usuarios_ids, itens_ids = usuarios_ids.flatten().long(), itens_ids.flatten().long()

    for i, modelo_cliente in enumerate(modelos_clientes):
        # print(f"=== Treinamento no Cliente {i + 1} ===")
        indices_nao_avaliados = (avaliacoes[i] == 0).nonzero(as_tuple=False).squeeze()

        # indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP]]
        indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:NR_ADVANTAGED_GROUP if i in G[1] else NR_DISADVANTAGED_GROUP]]
        novas_avaliacoes = torch.randint(1, 6, (NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP,)).float()
        modelos_clientes_nr.append((i, NR_ADVANTAGED_GROUP if i < NUMBER_ADVANTAGED_GROUP else NR_DISADVANTAGED_GROUP))

        avaliacoes_final[i, indices_novas_avaliacoes] = novas_avaliacoes
        avaliacoes_final_cliente = avaliacoes.clone()
        avaliacoes_final_cliente[i, indices_novas_avaliacoes] = novas_avaliacoes

        optimizer_cliente = optim.SGD(modelo_cliente.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer_cliente = optim.Adam(modelo_cliente.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        for epoch in range(epochs):
            optimizer_cliente.zero_grad()
            predictions = modelo_cliente(usuarios_ids, itens_ids).view(num_usuarios, num_itens).float()
            loss_cliente = criterion(predictions, avaliacoes_final_cliente)
            loss_cliente.backward()
            optimizer_cliente.step()

        with torch.no_grad():
            recomendacoes_cliente = modelo_cliente(usuarios_ids, itens_ids).view(num_usuarios, num_itens)

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

    return avaliacoes_final, modelos_clientes, modelos_clientes_rindv, modelos_clientes_loss, modelos_clientes_nr

def agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global, modelos_clientes):
    with torch.no_grad():
        for i, param_global in enumerate(modelo_global.parameters()):
            cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
            param_global.copy_(cliente_params.mean(dim=0))

def agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global, modelos_clientes, learning_rate=0.034):
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
    # print("total_perdas")
    # print(total_perdas)

    # Calcular os pesos de agregação baseados nas perdas
    pesos = [perda / total_perdas for _, perda in modelos_clientes_rindv]

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
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_loss")
    # print("modelos_clientes_loss")
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
    # Calcular o total_nr
    total_nr = sum(nr for _, nr in modelos_clientes_nr)
    # print("total_nr")
    # print(total_nr)

    # Calcular os pesos de agregação baseados nos valores inversos de total_nr
    pesos = [total_nr / nr if nr != 0 else 0 for _, nr in modelos_clientes_nr]

    # Normalizar os pesos para que a soma seja 1
    total_pesos = sum(pesos)
    pesos = [peso / total_pesos for peso in pesos]
    # print("\n\nagregar_modelos_locais_ao_global_media_poderada_pesos_nr")
    # print("modelos_clientes_nr")
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
        recomendacoes_tensor = modelo_global(usuarios_ids.long(), itens_ids.long()).view(num_usuarios, num_itens)

    recomendacoes_np = recomendacoes_tensor.numpy()
    recomendacoes_df = pd.DataFrame(recomendacoes_np)

    ilv = IndividualLossVariance(avaliacoes_df, omega, 1)

    algorithmImpartiality_01_ma_rindv_np = AlgorithmImpartiality(avaliacoes_df, omega, 1)
    list_X_est = algorithmImpartiality_01_ma_rindv_np.evaluate(recomendacoes_df, 3) # calculates a list of h estimated matrices => h = 3

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
    caminho_do_arquivo = 'X_MovieLens-1M.xlsx'
    avaliacoes_inicial_tensor, avaliacoes_inicial_df = carregar_avaliacoes_do_arquivo_xls(caminho_do_arquivo)

    with torch.no_grad():
        # Preparar os índices de usuários e itens
        num_usuarios, num_itens = avaliacoes_inicial_tensor.shape
        usuarios_ids, itens_ids = torch.meshgrid(torch.arange(num_usuarios), torch.arange(num_itens), indexing='ij')
        usuarios_ids = usuarios_ids.reshape(-1)
        itens_ids = itens_ids.reshape(-1)

        # Assegurando que os índices sejam inteiros longos para compatibilidade com embedding layers
        usuarios_ids_long = usuarios_ids.long()
        itens_ids_long = itens_ids.long()

    # Exemplo de uso
    # num_usuarios = len(usuarios_ids_long)
    # num_itens = len(itens_ids_long)
    embedding_size = 10
    hidden_size = 16

    modelo_global_federado_01_ma_tensor = RecommendationNN(num_usuarios, num_itens, embedding_size, hidden_size)
    criterion = nn.MSELoss() 

    # Realizando cópias do modelo global para cada uma das situações a serem analisadas
    treinar_modelo_global(modelo_global_federado_01_ma_tensor, avaliacoes_inicial_tensor, criterion)
    modelo_global_federado_02_mp_rindv_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_03_mp_loss_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_04_mp_nr_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)

    modelo_global_federado_05_ma_rindv_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_05_ma_loss_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_05_ma_nr_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_06_mp_rindv_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_07_mp_loss_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_federado_08_mp_nr_fairness_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)

    modelo_global_naofederado_09_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_naofederado_10_mp_rindv_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_naofederado_11_mp_loss_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)
    modelo_global_naofederado_12_mp_nr_tensor = copy.deepcopy(modelo_global_federado_01_ma_tensor)


    with torch.no_grad():
        recomendacoes_inicial_01_ma_tensor = modelo_global_federado_01_ma_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        # recomendacoes_inicial_02_mp_rindv_tensor = modelo_global_federado_02_mp_rindv_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        # recomendacoes_inicial_03_mp_loss_tensor = modelo_global_federado_03_mp_loss_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        # recomendacoes_inicial_04_mp_nr_tensor = modelo_global_federado_04_mp_nr_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)

        # recomendacoes_inicial_05_ma_rindv_fairness_tensor = modelo_global_federado_05_ma_rindv_fairness_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        # recomendacoes_inicial_06_mp_rindv_fairness_tensor = modelo_global_federado_06_mp_rindv_fairness_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        # recomendacoes_inicial_07_mp_loss_fairness_tensor = modelo_global_federado_07_mp_loss_fairness_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        # recomendacoes_inicial_08_mp_nr_fairness_tensor = modelo_global_federado_08_mp_nr_fairness_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)

    avaliacoes_final_01_ma_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_02_mp_rindv_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_03_mp_loss_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_04_mp_nr_tensor = avaliacoes_inicial_tensor

    avaliacoes_final_05_ma_rindv_fairness_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_05_ma_loss_fairness_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_05_ma_nr_fairness_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_06_mp_rindv_fairness_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_07_mp_loss_fairness_tensor = avaliacoes_inicial_tensor
    avaliacoes_final_08_mp_nr_fairness_tensor = avaliacoes_inicial_tensor

    avaliacoes_final_09_naofederado_tensor = avaliacoes_inicial_tensor
    # avaliacoes_final_10_mp_rindv_naofederado_tensor = avaliacoes_inicial_tensor
    # avaliacoes_final_11_mp_loss_naofederado_tensor = avaliacoes_inicial_tensor
    # avaliacoes_final_12_mp_nr_naofederado_tensor = avaliacoes_inicial_tensor

    for round in range(1):
        print(f"\n=== ROUND {round} ===")

        print("\n=== CLIENTES (ETAPA DE TREINAMENTOS LOCAIS) ===")
        print("treinar_modelos_locais :: modelos_clientes_01_ma_tensor")
        avaliacoes_final_01_ma_tensor, modelos_clientes_01_ma_tensor, calculos_modelos_clientes_01_ma_rindv, calculos_modelos_clientes_01_ma_loss, calculos_modelos_clientes_01_ma_nr = treinar_modelos_locais(modelo_global_federado_01_ma_tensor, avaliacoes_final_01_ma_tensor, criterion)

        print("treinar_modelos_locais :: modelos_clientes_02_mp_rindv_tensor")
        avaliacoes_final_02_mp_rindv_tensor, modelos_clientes_02_mp_rindv_tensor, calculos_modelos_clientes_02_mp_rindv_rindv, calculos_modelos_clientes_02_mp_rindv_loss, calculos_modelos_clientes_02_mp_rindv_nr = treinar_modelos_locais(modelo_global_federado_02_mp_rindv_tensor, avaliacoes_final_02_mp_rindv_tensor, criterion)
        print("treinar_modelos_locais :: modelos_clientes_03_mp_loss_tensor")
        avaliacoes_final_03_mp_loss_tensor, modelos_clientes_03_mp_loss_tensor, calculos_modelos_clientes_03_mp_loss_rindv, calculos_modelos_clientes_03_mp_loss_loss, calculos_modelos_clientes_03_mp_loss_nr = treinar_modelos_locais(modelo_global_federado_03_mp_loss_tensor, avaliacoes_final_03_mp_loss_tensor, criterion)
        print("treinar_modelos_locais :: modelos_clientes_04_mp_nr_tensor")
        avaliacoes_final_04_mp_nr_tensor, modelos_clientes_04_mp_nr_tensor, calculos_modelos_clientes_04_mp_nr_rindv, calculos_modelos_clientes_04_mp_nr_loss, calculos_modelos_clientes_04_mp_nr_nr = treinar_modelos_locais(modelo_global_federado_04_mp_nr_tensor, avaliacoes_final_04_mp_nr_tensor, criterion)

        print("treinar_modelos_locais :: modelos_clientes_05_ma_rindv_fairness_tensor")
        avaliacoes_final_05_ma_rindv_fairness_tensor, modelos_clientes_05_ma_rindv_fairness_tensor, calculos_modelos_clientes_05_ma_rindv_fairness_rindv, calculos_modelos_clientes_05_ma_loss_fairness_rindv, calculos_modelos_clientes_05_ma_nr_fairness_rindv = treinar_modelos_locais(modelo_global_federado_05_ma_rindv_fairness_tensor, avaliacoes_final_05_ma_rindv_fairness_tensor, criterion)
        print("treinar_modelos_locais :: modelos_clientes_05_ma_rindv_fairness_tensor")
        avaliacoes_final_05_ma_loss_fairness_tensor, modelos_clientes_05_ma_loss_fairness_tensor, calculos_modelos_clientes_05_ma_rindv_fairness_loss, calculos_modelos_clientes_05_ma_loss_fairness_loss, calculos_modelos_clientes_05_ma_nr_fairness_loss = treinar_modelos_locais(modelo_global_federado_05_ma_loss_fairness_tensor, avaliacoes_final_05_ma_loss_fairness_tensor, criterion)
        print("treinar_modelos_locais :: modelos_clientes_05_ma_rindv_fairness_tensor")
        avaliacoes_final_05_ma_nr_fairness_tensor, modelos_clientes_05_ma_nr_fairness_tensor, calculos_modelos_clientes_05_ma_rindv_fairness_nr, calculos_modelos_clientes_05_ma_loss_fairness_nr, calculos_modelos_clientes_05_ma_nr_fairness_nr = treinar_modelos_locais(modelo_global_federado_05_ma_nr_fairness_tensor, avaliacoes_final_05_ma_nr_fairness_tensor, criterion)


        print("\n=== SERVIDOR (ETAPA DE TREINAMENTO FINAL - AGREGAÇÃO) ===")
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado_01_ma_tensor")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado_01_ma_tensor, modelos_clientes_01_ma_tensor)
        # agregar_modelos_locais_ao_global_media_aritmetica_gradientes(modelo_global_federado1, modelos_clientes)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_rindv :: modelo_global_federado_02_mp_rindv_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(modelo_global_federado_02_mp_rindv_tensor, modelos_clientes_02_mp_rindv_tensor, calculos_modelos_clientes_02_mp_rindv_rindv)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_loss :: modelo_global_federado_03_mp_loss_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_loss(modelo_global_federado_03_mp_loss_tensor, modelos_clientes_03_mp_loss_tensor, calculos_modelos_clientes_03_mp_loss_loss)
        print("agregar_modelos_locais_ao_global_media_poderada_pesos_nr :: modelo_global_federado_04_mp_nr_tensor")
        agregar_modelos_locais_ao_global_media_poderada_pesos_nr(modelo_global_federado_04_mp_nr_tensor, modelos_clientes_04_mp_nr_tensor, calculos_modelos_clientes_04_mp_nr_nr)

        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado_05_ma_fairness_tensor")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado_05_ma_rindv_fairness_tensor, modelos_clientes_05_ma_rindv_fairness_tensor)
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado_05_ma_loss_fairness_tensor")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado_05_ma_loss_fairness_tensor, modelos_clientes_05_ma_loss_fairness_tensor)
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos :: modelo_global_federado_05_ma_nr_fairness_tensor")
        agregar_modelos_locais_ao_global_media_aritmetica_pesos(modelo_global_federado_05_ma_nr_fairness_tensor, modelos_clientes_05_ma_nr_fairness_tensor)

        # Agrupamento dos usuários no Sistema de Recomendação Federado
        users_modelos_clientes_05_ma_rindv_fairness_ordenados = sorted(calculos_modelos_clientes_05_ma_rindv_fairness_rindv, key=lambda x: x[1], reverse=False)
        list_users_05_ma_rindv_fairness = [i for i, _ in users_modelos_clientes_05_ma_rindv_fairness_ordenados]
        advantaged_group_05_ma_rindv_fairness = list_users_05_ma_rindv_fairness[0:15]
        disadvantaged_group_05_ma_rindv_fairness = list_users_05_ma_rindv_fairness[15:300]
        G_05_MA_RINDV = {1: advantaged_group_05_ma_rindv_fairness, 2: disadvantaged_group_05_ma_rindv_fairness}

        users_modelos_clientes_05_ma_loss_fairness_ordenados = sorted(calculos_modelos_clientes_05_ma_loss_fairness_loss, key=lambda x: x[1], reverse=False)
        list_users_05_ma_loss_fairness = [i for i, _ in users_modelos_clientes_05_ma_loss_fairness_ordenados]
        advantaged_group_05_ma_loss_fairness = list_users_05_ma_loss_fairness[0:15]
        disadvantaged_group_05_ma_loss_fairness = list_users_05_ma_loss_fairness[15:300]
        G_05_MA_LOSS = {1: advantaged_group_05_ma_loss_fairness, 2: disadvantaged_group_05_ma_loss_fairness}

        users_modelos_clientes_05_ma_nr_fairness_ordenados = sorted(calculos_modelos_clientes_05_ma_nr_fairness_nr, key=lambda x: x[1], reverse=True)
        list_users_05_ma_nr_fairness = [i for i, _ in users_modelos_clientes_05_ma_nr_fairness_ordenados]
        advantaged_group_05_ma_nr_fairness = list_users_05_ma_nr_fairness[0:15]
        disadvantaged_group_05_ma_nr_fairness = list_users_05_ma_nr_fairness[15:300]
        G_05_MA_NR = {1: advantaged_group_05_ma_nr_fairness, 2: disadvantaged_group_05_ma_nr_fairness}

        # Agrupamento dos usuários no Sistema de Recomendação Federado com Justiça
        # ??? Verificar se posso treinar o modelo baseado nas recomendações e não somente nas avaliações (como feito para os outros). Ou, ver se consigo gerar um modelo à partir dos tensores (mas de outra forma)
        recomendacoes_final_05_ma_rindv_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_05_ma_rindv_fairness_tensor, avaliacoes_final_05_ma_rindv_fairness_tensor, G_05_MA_RINDV)
        recomendacoes_final_05_ma_rindv_fairness_tensor = torch.tensor(recomendacoes_final_05_ma_rindv_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_05_ma_rindv_fairness_tensor, recomendacoes_final_05_ma_rindv_fairness_tensor, criterion)
        
        recomendacoes_final_05_ma_loss_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_05_ma_loss_fairness_tensor, avaliacoes_final_05_ma_loss_fairness_tensor, G_05_MA_LOSS)
        recomendacoes_final_05_ma_loss_fairness_tensor = torch.tensor(recomendacoes_final_05_ma_loss_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_05_ma_loss_fairness_tensor, recomendacoes_final_05_ma_loss_fairness_tensor, criterion)

        recomendacoes_final_05_ma_nr_fairness_df = aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global_federado_05_ma_nr_fairness_tensor, avaliacoes_final_05_ma_nr_fairness_tensor, G_05_MA_NR)
        recomendacoes_final_05_ma_nr_fairness_tensor = torch.tensor(recomendacoes_final_05_ma_nr_fairness_df.values, dtype=torch.float32)
        treinar_modelo_global(modelo_global_federado_05_ma_nr_fairness_tensor, recomendacoes_final_05_ma_nr_fairness_tensor, criterion)
        
        # Agrupamento dos usuários no Sistema de Recomendação NÃO Federado
        treinar_modelo_global(modelo_global_naofederado_09_tensor, avaliacoes_final_09_naofederado_tensor, criterion) # consideramos as avaliações geradas no treinamento dos modelos_clientes_01_ma_rindv_tensor 


    with torch.no_grad():
        recomendacoes_final_01_ma_tensor = modelo_global_federado_01_ma_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_02_mp_rindv_tensor = modelo_global_federado_02_mp_rindv_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_03_mp_loss_tensor = modelo_global_federado_03_mp_loss_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_04_mp_nr_tensor = modelo_global_federado_04_mp_nr_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_05_ma_rindv_fairness_tensor = modelo_global_federado_05_ma_rindv_fairness_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
        recomendacoes_final_09_naofederado_tensor = modelo_global_naofederado_09_tensor(usuarios_ids_long, itens_ids_long).view(num_usuarios, num_itens)
    
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
    avaliacoes_final_05_ma_rindv_fairness_np = avaliacoes_final_05_ma_rindv_fairness_tensor.numpy()
    avaliacoes_final_05_ma_rindv_fairness_df = pd.DataFrame(avaliacoes_final_05_ma_rindv_fairness_np)
    avaliacoes_final_05_ma_loss_fairness_np = avaliacoes_final_05_ma_loss_fairness_tensor.numpy()
    avaliacoes_final_05_ma_loss_fairness_df = pd.DataFrame(avaliacoes_final_05_ma_loss_fairness_np)
    avaliacoes_final_05_ma_nr_fairness_np = avaliacoes_final_05_ma_nr_fairness_tensor.numpy()
    avaliacoes_final_05_ma_nr_fairness_df = pd.DataFrame(avaliacoes_final_05_ma_nr_fairness_np)
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
    recomendacoes_final_05_ma_rindv_fairness_np = recomendacoes_final_05_ma_rindv_fairness_tensor.numpy()
    recomendacoes_final_05_ma_rindv_fairness_df = pd.DataFrame(recomendacoes_final_05_ma_rindv_fairness_np)
    recomendacoes_final_05_ma_loss_fairness_np = recomendacoes_final_05_ma_loss_fairness_tensor.numpy()
    recomendacoes_final_05_ma_loss_fairness_df = pd.DataFrame(recomendacoes_final_05_ma_loss_fairness_np)
    recomendacoes_final_05_ma_nr_fairness_np = recomendacoes_final_05_ma_nr_fairness_tensor.numpy()
    recomendacoes_final_05_ma_nr_fairness_df = pd.DataFrame(recomendacoes_final_05_ma_nr_fairness_np)
    recomendacoes_final_09_naofederado_np = recomendacoes_final_09_naofederado_tensor.numpy()
    recomendacoes_final_09_naofederado_df = pd.DataFrame(recomendacoes_final_09_naofederado_np)

    omega_inicial = (avaliacoes_inicial_df != 0)
    omega_final_01_ma = (avaliacoes_final_01_ma_df != 0)    
    omega_final_02_mp_rindv = (avaliacoes_final_02_mp_rindv_df != 0)    
    omega_final_03_mp_loss = (avaliacoes_final_03_mp_loss_df != 0)    
    omega_final_04_mp_nr = (avaliacoes_final_04_mp_nr_df != 0)    
    omega_final_05_ma_rindv_fairness = (avaliacoes_final_05_ma_rindv_fairness_df != 0)
    omega_final_05_ma_loss_fairness = (avaliacoes_final_05_ma_loss_fairness_df != 0)
    omega_final_05_ma_nr_fairness = (avaliacoes_final_05_ma_nr_fairness_df != 0)
    omega_final_09_naofederado = (avaliacoes_final_09_naofederado_df != 0)    

    # To capture polarization, we seek to measure the extent to which the user ratings disagree
    polarization = Polarization()
    Rpol_inicial = polarization.evaluate(recomendacoes_inicial_01_ma_df)
    Rpol_final_01_ma = polarization.evaluate(recomendacoes_final_01_ma_df)
    Rpol_final_02_mp_rindv = polarization.evaluate(recomendacoes_final_02_mp_rindv_df)
    Rpol_final_03_mp_loss = polarization.evaluate(recomendacoes_final_03_mp_loss_df)
    Rpol_final_04_mp_nr = polarization.evaluate(recomendacoes_final_04_mp_nr_df)
    Rpol_final_05_ma_rindv_fairness = polarization.evaluate(recomendacoes_final_05_ma_rindv_fairness_df)
    Rpol_final_05_ma_loss_fairness = polarization.evaluate(recomendacoes_final_05_ma_loss_fairness_df)
    Rpol_final_05_ma_nr_fairness = polarization.evaluate(recomendacoes_final_05_ma_nr_fairness_df)
    Rpol_final_09_naofederado = polarization.evaluate(recomendacoes_final_09_naofederado_df)
    print(f"\nPolarization Inicial (Rpol)                                        : {Rpol_inicial:.9f}")
    print(f"Polarization Final   (Rpol [1 :: Média Aritmética               ]) : {Rpol_final_01_ma:.9f}")
    print(f"Polarization Final   (Rpol [2 :: Média Ponderada Rindv          ]) : {Rpol_final_02_mp_rindv:.9f}")
    print(f"Polarization Final   (Rpol [3 :: Média Ponderada Loss           ]) : {Rpol_final_03_mp_loss:.9f}")
    print(f"Polarization Final   (Rpol [4 :: Média Ponderada NR             ]) : {Rpol_final_04_mp_nr:.9f}")
    print(f"Polarization Final   (Rpol [5 :: Média Aritmética Fairness Rindv]) : {Rpol_final_05_ma_rindv_fairness:.9f}")
    print(f"Polarization Final   (Rpol [5 :: Média Aritmética Fairness Loss ]) : {Rpol_final_05_ma_loss_fairness:.9f}")
    print(f"Polarization Final   (Rpol [5 :: Média Aritmética Fairness NR   ]) : {Rpol_final_05_ma_nr_fairness:.9f}")
    print(f"Polarization Final   (Rpol [9 :: Não Federado                   ]) : {Rpol_final_09_naofederado:.9f}")

    ilv_inicial = IndividualLossVariance(avaliacoes_inicial_df, omega_inicial, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_01_ma = IndividualLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_02_mp_rindv = IndividualLossVariance(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_03_mp_loss = IndividualLossVariance(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_04_mp_nr = IndividualLossVariance(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_05_ma_rindv_fairness = IndividualLossVariance(avaliacoes_final_05_ma_rindv_fairness_df, omega_final_05_ma_rindv_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_05_ma_loss_fairness = IndividualLossVariance(avaliacoes_final_05_ma_loss_fairness_df, omega_final_05_ma_loss_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_05_ma_nr_fairness = IndividualLossVariance(avaliacoes_final_05_ma_nr_fairness_df, omega_final_05_ma_nr_fairness, 1) #axis = 1 (0 rows e 1 columns)
    ilv_final_09_naofederado = IndividualLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, 1) #axis = 1 (0 rows e 1 columns)
    Rindv_inicial = ilv_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    Rindv_final_01_ma = ilv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    Rindv_final_02_mp_rindv = ilv_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    Rindv_final_03_mp_loss = ilv_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    Rindv_final_04_mp_nr = ilv_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    Rindv_final_05_ma_rindv_fairness = ilv_final_05_ma_rindv_fairness.evaluate(recomendacoes_final_05_ma_rindv_fairness_df)
    Rindv_final_05_ma_loss_fairness = ilv_final_05_ma_rindv_fairness.evaluate(recomendacoes_final_05_ma_loss_fairness_df)
    Rindv_final_05_ma_nr_fairness = ilv_final_05_ma_rindv_fairness.evaluate(recomendacoes_final_05_ma_nr_fairness_df)
    Rindv_final_09_naofederado = ilv_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)
    print(f"\nIndividual Loss Variance (Rindv Inicial)                                      : {Rindv_inicial:.9f}")
    print(f"Individual Loss Variance (Rindv Final [1 :: Média Aritmética               ]) : {Rindv_final_01_ma:.9f}")
    print(f"Individual Loss Variance (Rindv Final [2 :: Média Ponderada Rindv          ]) : {Rindv_final_02_mp_rindv:.9f}")
    print(f"Individual Loss Variance (Rindv Final [3 :: Média Ponderada Loss           ]) : {Rindv_final_03_mp_loss:.9f}")
    print(f"Individual Loss Variance (Rindv Final [4 :: Média Ponderada NR             ]) : {Rindv_final_04_mp_nr:.9f}")
    print(f"Individual Loss Variance (Rindv Final [5 :: Média Aritmética Rindv Fairness]) : {Rindv_final_05_ma_rindv_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [5 :: Média Aritmética Loss Fairness ]) : {Rindv_final_05_ma_loss_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [5 :: Média Aritmética NR Fairness   ]) : {Rindv_final_05_ma_nr_fairness:.9f}")
    print(f"Individual Loss Variance (Rindv Final [9 :: Não Federado                   ]) : {Rindv_final_09_naofederado:.9f}")

    # # # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # # # advantaged group: 5% users with the highest number of item ratings
    # # # disadvantaged group: 95% users with the lowest number of item ratings
    # modelos_clientes_rindv_ordenados = sorted(calculos_modelos_clientes_02_mp_rindv_rindv, key=lambda x: x[1], reverse=False)
    # list_users_rindv = [i for i, _ in modelos_clientes_rindv_ordenados]
    # advantaged_group_rindv = list_users_rindv[0:15]
    # disadvantaged_group_rindv = list_users_rindv[15:300]
    # G_RINDV = {1: advantaged_group_rindv, 2: disadvantaged_group_rindv}

    # modelos_clientes_loss_ordenados = sorted(calculos_modelos_clientes_03_mp_loss_loss, key=lambda x: x[1], reverse=False)
    # list_users_loss = [i for i, _ in modelos_clientes_loss_ordenados]
    # advantaged_group_loss = list_users_loss[0:15]
    # disadvantaged_group_loss = list_users_loss[15:300]
    # G_LOSS = {1: advantaged_group_loss, 2: disadvantaged_group_loss}
    
    # modelos_clientes_nr_ordenados = sorted(calculos_modelos_clientes_04_mp_nr_nr, key=lambda x: x[1], reverse=True)
    # list_users_nr = [i for i, _ in modelos_clientes_nr_ordenados]
    # advantaged_group_nr = list_users_nr[0:15]
    # disadvantaged_group_nr = list_users_nr[15:300]
    # G_NR = {1: advantaged_group_nr, 2: disadvantaged_group_nr}

    users_modelos_clientes_01_ma_rindv_ordenados = sorted(calculos_modelos_clientes_01_ma_rindv, key=lambda x: x[1], reverse=False)
    list_users_01_ma_rindv = [i for i, _ in users_modelos_clientes_01_ma_rindv_ordenados]
    advantaged_group_01_ma_rindv = list_users_01_ma_rindv[0:15]
    disadvantaged_group_01_ma_rindv = list_users_01_ma_rindv[15:300]
    G_01_MA_RINDV = {1: advantaged_group_01_ma_rindv, 2: disadvantaged_group_01_ma_rindv}

    users_modelos_clientes_01_ma_loss_ordenados = sorted(calculos_modelos_clientes_01_ma_loss, key=lambda x: x[1], reverse=False)
    list_users_01_ma_loss = [i for i, _ in users_modelos_clientes_01_ma_loss_ordenados]
    advantaged_group_01_ma_loss = list_users_01_ma_loss[0:15]
    disadvantaged_group_01_ma_loss = list_users_01_ma_loss[15:300]
    G_01_MA_LOSS = {1: advantaged_group_01_ma_loss, 2: disadvantaged_group_01_ma_loss}

    users_modelos_clientes_01_ma_nr_ordenados = sorted(calculos_modelos_clientes_01_ma_nr, key=lambda x: x[1], reverse=True)
    list_users_01_ma_nr = [i for i, _ in users_modelos_clientes_01_ma_nr_ordenados]
    advantaged_group_01_ma_nr = list_users_01_ma_nr[0:15]
    disadvantaged_group_01_ma_nr = list_users_01_ma_nr[15:300]
    G_01_MA_NR = {1: advantaged_group_01_ma_nr, 2: disadvantaged_group_01_ma_nr}

    users_modelos_clientes_02_mp_rindv_rindv_ordenados = sorted(calculos_modelos_clientes_02_mp_rindv_rindv, key=lambda x: x[1], reverse=False)
    list_users_02_mp_rindv_rindv = [i for i, _ in users_modelos_clientes_02_mp_rindv_rindv_ordenados]
    advantaged_group_02_mp_rindv_rindv = list_users_02_mp_rindv_rindv[0:15]
    disadvantaged_group_02_mp_rindv_rindv = list_users_02_mp_rindv_rindv[15:300]
    G_02_MP_RINDV = {1: advantaged_group_02_mp_rindv_rindv, 2: disadvantaged_group_02_mp_rindv_rindv}

    users_modelos_clientes_03_mp_loss_loss_ordenados = sorted(calculos_modelos_clientes_03_mp_loss_loss, key=lambda x: x[1], reverse=False)
    list_users_03_mp_loss_loss = [i for i, _ in users_modelos_clientes_03_mp_loss_loss_ordenados]
    advantaged_group_03_mp_loss_loss = list_users_03_mp_loss_loss[0:15]
    disadvantaged_group_03_mp_loss_loss = list_users_03_mp_loss_loss[15:300]
    G_03_MP_LOSS = {1: advantaged_group_03_mp_loss_loss, 2: disadvantaged_group_03_mp_loss_loss}

    users_modelos_clientes_04_mp_nr_nr_ordenados = sorted(calculos_modelos_clientes_04_mp_nr_nr, key=lambda x: x[1], reverse=True)
    list_users_04_mp_nr_nr = [i for i, _ in users_modelos_clientes_04_mp_nr_nr_ordenados]
    advantaged_group_04_mp_nr_nr = list_users_04_mp_nr_nr[0:15]
    disadvantaged_group_04_mp_nr_nr = list_users_04_mp_nr_nr[15:300]
    G_04_MP_NR = {1: advantaged_group_04_mp_nr_nr, 2: disadvantaged_group_04_mp_nr_nr}

    users_modelos_clientes_05_ma_rindv_fairness_ordenados = sorted(calculos_modelos_clientes_05_ma_rindv_fairness_rindv, key=lambda x: x[1], reverse=False)
    list_users_05_ma_rindv_fairness = [i for i, _ in users_modelos_clientes_05_ma_rindv_fairness_ordenados]
    advantaged_group_05_ma_rindv_fairness = list_users_05_ma_rindv_fairness[0:15]
    disadvantaged_group_05_ma_rindv_fairness = list_users_05_ma_rindv_fairness[15:300]
    G_05_MA_RINDV = {1: advantaged_group_05_ma_rindv_fairness, 2: disadvantaged_group_05_ma_rindv_fairness}

    users_modelos_clientes_05_ma_loss_fairness_ordenados = sorted(calculos_modelos_clientes_05_ma_loss_fairness_loss, key=lambda x: x[1], reverse=False)
    list_users_05_ma_loss_fairness = [i for i, _ in users_modelos_clientes_05_ma_loss_fairness_ordenados]
    advantaged_group_05_ma_loss_fairness = list_users_05_ma_loss_fairness[0:15]
    disadvantaged_group_05_ma_loss_fairness = list_users_05_ma_loss_fairness[15:300]
    G_05_MA_LOSS = {1: advantaged_group_05_ma_loss_fairness, 2: disadvantaged_group_05_ma_loss_fairness}

    users_modelos_clientes_05_ma_nr_fairness_ordenados = sorted(calculos_modelos_clientes_05_ma_nr_fairness_nr, key=lambda x: x[1], reverse=True)
    list_users_05_ma_nr_fairness = [i for i, _ in users_modelos_clientes_05_ma_nr_fairness_ordenados]
    advantaged_group_05_ma_nr_fairness = list_users_05_ma_nr_fairness[0:15]
    disadvantaged_group_05_ma_nr_fairness = list_users_05_ma_nr_fairness[15:300]
    G_05_MA_NR = {1: advantaged_group_05_ma_nr_fairness, 2: disadvantaged_group_05_ma_nr_fairness}

    users_modelos_clientes_09_naofederado_ordenados = sorted(calculos_modelos_clientes_01_ma_rindv, key=lambda x: x[1], reverse=False)
    list_users_09_naofederado = [i for i, _ in users_modelos_clientes_09_naofederado_ordenados]
    advantaged_group_09_naofederado = list_users_09_naofederado[0:15]
    disadvantaged_group_09_naofederado = list_users_09_naofederado[15:300]
    G_09_NAOFEDER = {1: advantaged_group_09_naofederado, 2: disadvantaged_group_09_naofederado}

    print("G_01_MA_RINDV")
    print(G_01_MA_RINDV[1])

    print("G_02_MP_RINDV")
    print(G_02_MP_RINDV[1])

    print("G_03_MP_LOSS")
    print(G_03_MP_LOSS[1])

    print("G_04_MP_NR")
    print(G_04_MP_NR[1])

    print("G_05_MA_RINDV")
    print(G_05_MA_RINDV[1])

    print("G_05_MA_LOSS")
    print(G_05_MA_LOSS[1])

    print("G_05_MA_NR")
    print(G_05_MA_NR[1])

    print("G_09_NAOFEDER")
    print(G_09_NAOFEDER[1])

    glv_inicial = GroupLossVariance(avaliacoes_inicial_df, omega_inicial, G_01_MA_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_01_ma_rindv = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G_01_MA_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_01_ma_loss = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G_01_MA_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_01_ma_nr = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G_01_MA_NR, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_02_mp_rindv = GroupLossVariance(avaliacoes_final_02_mp_rindv_df, omega_final_02_mp_rindv, G_02_MP_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_03_mp_loss = GroupLossVariance(avaliacoes_final_03_mp_loss_df, omega_final_03_mp_loss, G_03_MP_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_04_mp_nr = GroupLossVariance(avaliacoes_final_04_mp_nr_df, omega_final_04_mp_nr, G_04_MP_NR, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_05_ma_fairness_rindv = GroupLossVariance(avaliacoes_final_05_ma_rindv_fairness_df, omega_final_05_ma_rindv_fairness, G_05_MA_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_05_ma_fairness_loss = GroupLossVariance(avaliacoes_final_05_ma_rindv_fairness_df, omega_final_05_ma_rindv_fairness, G_05_MA_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_05_ma_fairness_nr = GroupLossVariance(avaliacoes_final_05_ma_rindv_fairness_df, omega_final_05_ma_rindv_fairness, G_05_MA_NR, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_09_naofederado_rindv = GroupLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, G_01_MA_RINDV, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_09_naofederado_loss = GroupLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, G_01_MA_LOSS, 1) #axis = 1 (0 rows e 1 columns)
    glv_final_09_naofederado_nr = GroupLossVariance(avaliacoes_final_09_naofederado_df, omega_final_09_naofederado, G_01_MA_NR, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR_inicial = glv_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    RgrpNR_final_01_ma_rindv = glv_final_01_ma_rindv.evaluate(recomendacoes_final_01_ma_df)
    RgrpNR_final_01_ma_loss = glv_final_01_ma_loss.evaluate(recomendacoes_final_01_ma_df)
    RgrpNR_final_01_ma_nr = glv_final_01_ma_nr.evaluate(recomendacoes_final_01_ma_df)
    RgrpNR_final_02_mp_rindv = glv_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    RgrpNR_final_03_mp_loss = glv_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    RgrpNR_final_04_mp_nr = glv_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    RgrpNR_final_05_ma_fairness_rindv = glv_final_05_ma_fairness_rindv.evaluate(recomendacoes_final_05_ma_rindv_fairness_df)
    RgrpNR_final_05_ma_fairness_loss = glv_final_05_ma_fairness_loss.evaluate(recomendacoes_final_05_ma_rindv_fairness_df)
    RgrpNR_final_05_ma_fairness_nr = glv_final_05_ma_fairness_nr.evaluate(recomendacoes_final_05_ma_rindv_fairness_df)
    RgrpNR_final_09_naofederado_rindv = glv_final_09_naofederado_rindv.evaluate(recomendacoes_final_09_naofederado_df)
    RgrpNR_final_09_naofederado_loss = glv_final_09_naofederado_loss.evaluate(recomendacoes_final_09_naofederado_df)
    RgrpNR_final_09_naofederado_nr = glv_final_09_naofederado_nr.evaluate(recomendacoes_final_09_naofederado_df)
    print(f"\nGroup Loss Variance (Rgrp Inicial)                                      : {RgrpNR_inicial:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética Rindv         ]) : {RgrpNR_final_01_ma_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética Loss          ]) : {RgrpNR_final_01_ma_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética NR            ]) : {RgrpNR_final_01_ma_nr:.9f}")
    print(f"Group Loss Variance (Rgrp Final [2 :: Média Ponderada Rindv          ]) : {RgrpNR_final_02_mp_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [3 :: Média Ponderada Loss           ]) : {RgrpNR_final_03_mp_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [4 :: Média Ponderada NR             ]) : {RgrpNR_final_04_mp_nr:.9f}")
    print(f"Group Loss Variance (Rgrp Final [5 :: Média Aritmética Rindv Fairness]) : {RgrpNR_final_05_ma_fairness_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [5 :: Média Aritmética Loss Fairness ]) : {RgrpNR_final_05_ma_fairness_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [5 :: Média Aritmética NR Fairness   ]) : {RgrpNR_final_05_ma_fairness_nr:.9f}")
    print(f"Group Loss Variance (Rgrp Final [9 :: Não Federado :: Rindv          ]) : {RgrpNR_final_09_naofederado_rindv:.9f}")
    print(f"Group Loss Variance (Rgrp Final [9 :: Não Federado :: Loss           ]) : {RgrpNR_final_09_naofederado_loss:.9f}")
    print(f"Group Loss Variance (Rgrp Final [9 :: Não Federado :: NR             ]) : {RgrpNR_final_09_naofederado_nr:.9f}")

    rmse_inicial = RMSE(avaliacoes_inicial_df, omega_inicial)
    result_inicial = rmse_inicial.evaluate(recomendacoes_inicial_01_ma_df)
    rmse_final_01_ma = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)
    result_final_01_ma = rmse_final_01_ma.evaluate(recomendacoes_final_01_ma_df)
    rmse_final_02_mp_rindv = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)
    result_final_02_mp_rindv = rmse_final_02_mp_rindv.evaluate(recomendacoes_final_02_mp_rindv_df)
    rmse_final_03_mp_loss = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)
    result_final_03_mp_loss = rmse_final_03_mp_loss.evaluate(recomendacoes_final_03_mp_loss_df)
    rmse_final_04_mp_nr = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)
    result_final_04_mp_nr = rmse_final_04_mp_nr.evaluate(recomendacoes_final_04_mp_nr_df)
    rmse_final_05_ma_fairness = RMSE(avaliacoes_final_05_ma_rindv_fairness_df, omega_final_05_ma_rindv_fairness)
    result_final_05_ma_fairness = rmse_final_05_ma_fairness.evaluate(recomendacoes_final_01_ma_df)
    rmse_final_09_naofederado = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)
    result_final_09_naofederado = rmse_final_09_naofederado.evaluate(recomendacoes_final_09_naofederado_df)
    print(f'\nRMSE Inicial                                : {result_inicial:.9f}')
    print(f'RMSE Final [1 :: Média Aritmética         ] : {result_final_01_ma:.9f}')
    print(f'RMSE Final [2 :: Média Ponderada Rindv    ] : {result_final_02_mp_rindv:.9f}')
    print(f'RMSE Final [3 :: Média Ponderada Loss     ] : {result_final_03_mp_loss:.9f}')
    print(f'RMSE Final [4 :: Média Ponderada NR       ] : {result_final_04_mp_nr:.9f}')
    print(f'RMSE Final [5 :: Média Aritmética Fairness] : {result_final_05_ma_fairness:.9f}')
    print(f'RMSE Final [Não Federado                  ] : {result_final_09_naofederado:.9f}')

    # avaliacoes_inicial_df.to_excel("avaliacoes_inicial.xlsx", index=False)
    # avaliacoes_final_df.to_excel("avaliacoes_final.xlsx", index=False)
    # recomendacoes_inicial_df.to_excel("recomendacoes_inicial.xlsx", index=False)
    # recomendacoes_final_df1.to_excel("recomendacoes_final_df1.xlsx", index=False)
    # recomendacoes_final_df2.to_excel("recomendacoes_final_df2.xlsx", index=False)
    # recomendacoes_final_df3.to_excel("recomendacoes_final_df3.xlsx", index=False)
    # recomendacoes_final_df4.to_excel("recomendacoes_final_df4.xlsx", index=False)
    # recomendacoes_modelo_global_nao_federado_df.to_excel("recomendacoes_modelo_global_nao_federado_df.xlsx", index=False)

if __name__ == "__main__":
    main()
