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
    
class ClienteFedRecSys:

    def __init__(self, id, modelo_global, avaliacoes_locais):
        self.id = id
        self.modelo = modelo_global
        self.avaliacoes_locais_tensor = avaliacoes_locais
        self.modelo_loss = None
        self.modelo_rindv = None
        
    # id: identificador da linha do usuário (cliente) na matriz
    # quantidade: número de novas avaliações geradas
    # aleatorio: indica se as avaliações serão aleatórias ou fixas
    def adicionar_novas_avaliacoes(self, quantidade, aleatorio=False):
        if aleatorio:
            avaliacoes = [random.randint(1, 5) for _ in range(20)] # gerando uma lista de 20 avaliações aleatórias
        else:
            avaliacoes = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5] # gerando uma lista de 20 avaliações fixas

        avaliacoes_tensor = torch.tensor(avaliacoes)
        avaliacoes_tensor = avaliacoes_tensor.float()

        indices_nao_avaliados = (self.avaliacoes_locais_tensor == 0).nonzero(as_tuple=False).squeeze()
        indices_novas_avaliacoes = indices_nao_avaliados[torch.randperm(len(indices_nao_avaliados))[:quantidade]]
        novas_avaliacoes = avaliacoes_tensor[0:quantidade]

        self.avaliacoes_locais_tensor[indices_novas_avaliacoes] = novas_avaliacoes


    def treinar_modelo(self, epochs=5, learning_rate=0.02):
        optimizer_cliente = optim.SGD(self.modelo.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.modelo.train()

        recomendacoes_locais_tensor = None
        loss_cliente = None
        for e in range(epochs):
            optimizer_cliente.zero_grad()
            recomendacoes_locais_tensor = self.modelo(self.avaliacoes_locais_tensor)
            loss_cliente = criterion(recomendacoes_locais_tensor, self.avaliacoes_locais_tensor)
            loss_cliente.backward()
            optimizer_cliente.step()
            #print(f' Época: {e}; Perda: {loss_cliente}')

        avaliacoes_locais_np = self.avaliacoes_locais_tensor.numpy()
        avaliacoes_locais_df = pd.DataFrame(avaliacoes_locais_np)
        recomendacoes_locais_np = recomendacoes_locais_tensor.detach().numpy()
        recomendacoes_locais_df = pd.DataFrame(recomendacoes_locais_np)
        omega_avaliacoes_locais_df = (recomendacoes_locais_df != 0)
        ilv_cliente = IndividualLossVariance(avaliacoes_locais_df, omega_avaliacoes_locais_df, 1)
        rindv_cliente = ilv_cliente.evaluate(recomendacoes_locais_df)

        # print("avaliacoes_locais_df")
        # print(avaliacoes_locais_df)

        # print("recomendacoes_locais_df")
        # print(recomendacoes_locais_df)

        self.modelo_loss = loss_cliente.detach().item()
        self.modelo_rindv = rindv_cliente

class ServidorFedRecSys:

    def __init__(self):
        self.modelo_global = None
        self.modelos_locais = []
        self.modelos_locais_loss = []
        self.modelos_locais_rindv = []
        self.numero_de_usuarios = None
        self.numero_de_itens = None
        self.avaliacoes_inicial = None
        self.avaliacoes_inicial_tensor = None
        self.avaliacoes_final_tensor = None # Este atributo foi adicionado apenas para comparar a injustiça nos métodos de agregação. Em um servidor real, não deveria existir.
    
    def iniciar_modelo(self, arquivo_excel):
        df = pd.read_excel(arquivo_excel) # Carregar os dados do arquivo Excel para um DataFrame do pandas
        df_com_zero = df.fillna(0) # Substituir valores NaN por zero
        df_dados = df_com_zero.iloc[:, 1:] # Selecionar apenas as colunas de dados com as avaliações dos filmes
        tensor_dados = torch.tensor(df_dados.values, dtype=torch.float32) # Converter o DataFrame para um tensor PyTorch
        df_dados.reset_index(drop=True)

        self.numero_de_usuarios = tensor_dados.shape[0]
        self.numero_de_itens = tensor_dados.shape[1]
        self.avaliacoes_inicial = df_dados
        self.avaliacoes_inicial_tensor = tensor_dados

        print("self.avaliacoes_inicial")
        print(self.avaliacoes_inicial)

        self.modelo_global = SimpleNN(self.numero_de_itens, 20, self.numero_de_itens)

    def treinar_modelo(self, epochs=5, learning_rate=0.02):
        optimizer_servidor = optim.SGD(self.modelo.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.modelo.train()

        recomendacoes_globais_tensor = None
        loss_servidor = None
        for e in range(epochs):
            optimizer_servidor.zero_grad()
            recomendacoes_globais_tensor = self.modelo(self.avaliacoes_final_tensor)
            loss_servidor = criterion(recomendacoes_globais_tensor, self.avaliacoes_final_tensor)
            loss_servidor.backward()
            optimizer_servidor.step()
            #print(f' Época: {e}; Perda: {loss_cliente}')

    def adicionar_avaliacoes_cliente(self, cliente_tensor):
        if self.avaliacoes_final_tensor is None:
            self.avaliacoes_final_tensor = cliente_tensor.unsqueeze(0)  # Cria um novo tensor com uma dimensão extra
        else:
            self.avaliacoes_final_tensor = torch.cat((self.avaliacoes_final_tensor, cliente_tensor.unsqueeze(0)), dim=0)


    def agregar_modelos_locais_ao_global_media_aritmetica_pesos(self, modelos_clientes):
        with torch.no_grad():
            for i, param_global in enumerate(self.modelo_global.parameters()):
                cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
                param_global.copy_(cliente_params.mean(dim=0))


    def agregar_modelos_locais_ao_global_media_poderada_pesos_loss(self, modelos_clientes, modelos_clientes_loss):

        # print("modelos_clientes_loss")
        # print(modelos_clientes_loss)

        # Calcular o total de perdas dos modelos locais (loss)
        total_perdas = sum(modelos_clientes_loss)

        # Calcular os pesos de agregação baseados nas perdas (loss)
        pesos = [perda / total_perdas for perda in modelos_clientes_loss]

        # Atualizar os parâmetros do modelo global com a média ponderada
        with torch.no_grad():
            for i, param_global in enumerate(self.modelo_global.parameters()):
                param_medio = torch.zeros_like(param_global)
                for j, peso in enumerate(pesos):
                    cliente_params = list(modelos_clientes[j].parameters())[i].data
                    param_medio += peso * cliente_params
                param_global.copy_(param_medio)

    
    def agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(self, modelos_clientes, modelos_clientes_rindv):

        # print("modelos_clientes_rindv")
        # print(modelos_clientes_rindv)

        # Calcular o total das injustiças individuais (Rindv)
        total_rindv = sum(modelos_clientes_rindv)

        # Calcular os pesos de agregação baseados nas rindv's
        pesos = [rindv / total_rindv for rindv in modelos_clientes_rindv]

        # Atualizar os parâmetros do modelo global com a média ponderada
        with torch.no_grad():
            for i, param_global in enumerate(self.modelo_global.parameters()):
                param_medio = torch.zeros_like(param_global)
                for j, peso in enumerate(pesos):
                    cliente_params = list(modelos_clientes[j].parameters())[i].data
                    param_medio += peso * cliente_params
                param_global.copy_(param_medio)

    
    # def aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(modelo_global, modelos_clientes_rindv, G):
    
    #     avaliacoes_np = avaliacoes.numpy()
    #     avaliacoes_df = pd.DataFrame(avaliacoes_np)

    #     omega = (avaliacoes_df != 0)

    #     num_usuarios, num_itens = avaliacoes.shape
    #     with torch.no_grad():
    #         recomendacoes_tensor = modelo_global(avaliacoes)

    #     recomendacoes_np = recomendacoes_tensor.numpy()
    #     recomendacoes_df = pd.DataFrame(recomendacoes_np)

    #     ilv = IndividualLossVariance(avaliacoes_df, omega, 1)

    #     algorithmImpartiality_01_ma_np = AlgorithmImpartiality(avaliacoes_df, omega, 1)
    #     list_X_est = algorithmImpartiality_01_ma_np.evaluate(recomendacoes_df, 5) # calculates a list of h estimated matrices => h = 5

    #     list_losses = []
    #     for X_est in list_X_est:
    #         losses = ilv.get_losses(X_est)
    #         list_losses.append(losses)

    #     Z = AlgorithmImpartiality.losses_to_Z(list_losses, num_usuarios)
    #     list_Zs = AlgorithmImpartiality.matrices_Zs(Z, G)
    #     recomendacoes_fairness_np = AlgorithmImpartiality.make_matrix_X_gurobi(list_X_est, G, list_Zs) # recomendações com justiça
    #     return recomendacoes_fairness_np


def iniciar_FedFairRecSys (dataset, G, rounds = 1, epochs=5, learning_rate=0.02, metodo_agregacao = 'ma'):

    print(f"\nMÉTODO DE AGREGAÇÃO :: {metodo_agregacao}")

    servidor = ServidorFedRecSys()
    servidor.iniciar_modelo(dataset)

    print(f"INSTANCIANDO CLIENTES LOCAIS")
    clientes = []
    for i in range(servidor.numero_de_usuarios):
        # print(f"Cliente {i} :: Instanciando")
        cliente = ClienteFedRecSys(i, servidor.modelo_global, servidor.avaliacoes_inicial_tensor[i])
        clientes.append(cliente)

    print(f"TREINANDO CLIENTES LOCAIS")
    for round in range (rounds):
        print(f"\nRound: {round}")

        servidor.modelos_locais = []
        servidor.modelos_locais_loss = []
        servidor.modelos_locais_rindv = []
        servidor.avaliacoes_final_tensor = None
        
        for cliente in clientes:
            # print(f"Cliente {cliente.id} :: Adicionando Avaliações e Treinando")

            # cliente.adicionar_novas_avaliacoes(1, False)

            if cliente.id < 15:
                cliente.adicionar_novas_avaliacoes(10, False)
            else:
                cliente.adicionar_novas_avaliacoes(1, False)
            cliente.treinar_modelo(epochs, learning_rate)

            # print(f"cliente.modelo_loss {cliente.modelo_loss}")
            # print(f"cliente.modelo_rindv {cliente.modelo_rindv}")

            servidor.modelos_locais.append(cliente.modelo)
            servidor.modelos_locais_loss.append(cliente.modelo_loss)
            servidor.modelos_locais_rindv.append(cliente.modelo_rindv)
            servidor.adicionar_avaliacoes_cliente(cliente.avaliacoes_locais_tensor.clone())
            
            # print(cliente.avaliacoes_locais)
        if metodo_agregacao == 'ma':
            servidor.agregar_modelos_locais_ao_global_media_aritmetica_pesos(servidor.modelos_locais)
        elif metodo_agregacao == 'mp_loss':
            servidor.agregar_modelos_locais_ao_global_media_poderada_pesos_loss(servidor.modelos_locais, servidor.modelos_locais_loss)
        elif metodo_agregacao == 'mp_rindv':
            servidor.agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(servidor.modelos_locais, servidor.modelos_locais_rindv)

    with torch.no_grad():
            recomendacoes_tensor = servidor.modelo_global(servidor.avaliacoes_final_tensor)

            

    # print("servidor.avaliacoes_final_tensor")
    # print(servidor.avaliacoes_final_tensor)

    # print("recomendacoes_final_01_ma_tensor")
    # print(recomendacoes_final_01_ma_tensor)

    print("=== MEDIDAS DE JUSTIÇA ===")

    avaliacoes = servidor.avaliacoes_final_tensor.numpy()
    avaliacoes_df = pd.DataFrame(avaliacoes)

    recomendacoes_np = recomendacoes_tensor.numpy()
    recomendacoes_df = pd.DataFrame(recomendacoes_np)

    omega = (avaliacoes_df != 0)  

    polarization = Polarization()
    Rpol = polarization.evaluate(recomendacoes_df)

    ilv = IndividualLossVariance(avaliacoes_df, omega, 1) #axis = 1 (0 rows e 1 columns)
    Rindv = ilv.evaluate(recomendacoes_df)

    glv = GroupLossVariance(avaliacoes_df, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    Rgrp = glv.evaluate(recomendacoes_df)

    rmse = RMSE(avaliacoes_df, omega)
    result_rmse = rmse.evaluate(recomendacoes_df)

    print(f"Polarization (Rpol) : {Rpol:.9f}")
    print(f"Individual Loss Variance (Rindv) : {Rindv:.9f}")
    print(f"Group Loss Variance (Rgrp) : {Rgrp:.9f}")
    print(f'RMSE : {result_rmse:.9f}')

    avaliacoes_df.to_excel(f"{dataset}-avaliacoes_df-{metodo_agregacao}.xlsx", index=False)
    recomendacoes_df.to_excel(f"{dataset}-recomendacoes_df-{metodo_agregacao}.xlsx", index=False)



# dataset='X-u5-i10.xlsx'
# G = {1: list(range(0, 2)), 2: list(range(2, 5))}

dataset='X_MovieLens-1M.xlsx'
G = {1: list(range(0, 15)), 2: list(range(15, 300))}
rounds=5
epochs=1000
learning_rate=0.02
print(f"\nFedFairRecSys")
iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, metodo_agregacao='ma')
iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, metodo_agregacao='mp_loss')
iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, metodo_agregacao='mp_rindv')



    