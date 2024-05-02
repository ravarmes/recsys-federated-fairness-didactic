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
    
class ClienteSistemaRecomendacao:

    def __init__(self, id, modelo_global, avaliacoes_locais):
        self.id = id
        self.modelo = modelo_global
        self.avaliacoes_locais_tensor = avaliacoes_locais
        self.modelo_perda = None
        
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


    def treinar_localmente(self, epochs=5, learning_rate=0.02):
        optimizer_cliente = optim.SGD(self.modelo.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.modelo.train()

        for e in range(epochs):
            optimizer_cliente.zero_grad()
            output_cliente = self.modelo(self.avaliacoes_locais_tensor)
            loss_cliente = criterion(output_cliente, self.avaliacoes_locais_tensor)
            loss_cliente.backward()
            optimizer_cliente.step()
            self.modelo_perda = loss_cliente
            #print(f' Época: {e}; Perda: {loss_cliente}')

class ServidorSistemaRecomendacao:

    def __init__(self):
        self.modelo_global = None
        self.modelos_locais = []
        self.numero_de_usuarios = None
        self.numero_de_itens = None
        self.avaliacoes_inicial = None
        self.avaliacoes_inicial_tensor = None
        self.avaliacoes_final_tensor = None
    
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

        self.modelo_global = SimpleNN(self.numero_de_itens, 20, self.numero_de_itens)


    def adicionar_avaliacoes_cliente(self, cliente_tensor):
        if self.avaliacoes_final_tensor is None:
            self.avaliacoes_final_tensor = cliente_tensor.unsqueeze(0)  # Cria um novo tensor com uma dimensão extra
        else:
            self.avaliacoes_final_tensor = torch.cat((self.avaliacoes_final_tensor, cliente_tensor.unsqueeze(0)), dim=0)


    def agregar_modelos_locais_ao_global_media_aritmetica_pesos(self, modelos_clientes):
        print("agregar_modelos_locais_ao_global_media_aritmetica_pesos")
        with torch.no_grad():
            for i, param_global in enumerate(self.modelo_global.parameters()):
                cliente_params = torch.stack([list(cliente.parameters())[i].data for cliente in modelos_clientes])
                param_global.copy_(cliente_params.mean(dim=0))

    


servidor = ServidorSistemaRecomendacao()
servidor.iniciar_modelo('X-u5-i10.xlsx') #  X_MovieLens-1M.xlsx



print(f"INSTANCIANDO CLIENTES LOCAIS")
clientes = []
for i in range(servidor.numero_de_usuarios):
    print(f"Cliente {i} :: Instanciando")
    cliente = ClienteSistemaRecomendacao(i, servidor.modelo_global, servidor.avaliacoes_inicial_tensor[i])
    clientes.append(cliente)

print(f"\nTREINANDO CLIENTES LOCAIS")
for round in range (2):
    servidor.avaliacoes_final_tensor = None
    print(f"\nRound: {round}")
    for cliente in clientes:
        print(f"Cliente {cliente.id} :: Adicionando Avaliações e Treinando")
        cliente.adicionar_novas_avaliacoes(1, False)
        cliente.treinar_localmente(epochs=50)
        servidor.modelos_locais.append(cliente.modelo)
        servidor.adicionar_avaliacoes_cliente(cliente.avaliacoes_locais_tensor.clone())
        
        # print(cliente.avaliacoes_locais)

    servidor.agregar_modelos_locais_ao_global_media_aritmetica_pesos(servidor.modelos_locais)

with torch.no_grad():
        recomendacoes_final_01_ma_tensor = servidor.modelo_global(servidor.avaliacoes_final_tensor)

# print("servidor.avaliacoes_final_tensor")
# print(servidor.avaliacoes_final_tensor)

# print("recomendacoes_final_01_ma_tensor")
# print(recomendacoes_final_01_ma_tensor)

print("\n=== MEDIDA DE JUSTIÇA ===")
G = {1: list(range(0, 2)), 2: list(range(2, 5))} # NR (Number Ratings)
print(G)

avaliacoes_final_01_ma_np = servidor.avaliacoes_final_tensor.numpy()
avaliacoes_final_01_ma_df = pd.DataFrame(avaliacoes_final_01_ma_np)

recomendacoes_final_01_ma_np = recomendacoes_final_01_ma_tensor.numpy()
recomendacoes_final_01_ma_df = pd.DataFrame(recomendacoes_final_01_ma_np)

omega_final_01_ma = (avaliacoes_final_01_ma_df != 0)  

polarization = Polarization()
Rpol_final_01_ma = polarization.evaluate(recomendacoes_final_01_ma_df)

print(f"Polarization Final   (Rpol [1 :: Média Aritmética                          ]) : {Rpol_final_01_ma:.9f}")

ilv_final_01_ma = IndividualLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, 1) #axis = 1 (0 rows e 1 columns)

Rindv_final_01_ma = ilv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)

print(f"Individual Loss Variance (Rindv Final [1 :: Média Aritmética               ]) : {Rindv_final_01_ma:.9f}")

glv_final_01_ma = GroupLossVariance(avaliacoes_final_01_ma_df, omega_final_01_ma, G, 1) #axis = 1 (0 rows e 1 columns)

RgrpNR_final_01_ma = glv_final_01_ma.evaluate(recomendacoes_final_01_ma_df)

print(f"Group Loss Variance (Rgrp Final [1 :: Média Aritmética                     ]) : {RgrpNR_final_01_ma:.9f}")

rmse_final_01_ma = RMSE(avaliacoes_final_01_ma_df, omega_final_01_ma)

result_final_01_ma = rmse_final_01_ma.evaluate(recomendacoes_final_01_ma_df)

print(f'RMSE Final [1 :: Média Aritmética                                          ]) : {result_final_01_ma:.9f}')

print("Fim")

    