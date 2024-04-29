import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance
from AlgorithmImpartiality import AlgorithmImpartiality
import random


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MatrixFactorizationNN(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorizationNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(0.001))
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(0.001))

    def call(self, inputs):
        user_id = inputs[:, 0]
        item_id = inputs[:, 1]
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        interaction = tf.multiply(user_vec, item_vec)
        score = tf.reduce_sum(interaction, axis=1)
        predicted_ratings = 1 + 4.0 * tf.sigmoid(score)
        return predicted_ratings

    def predict_all(self):
        user_ids = np.repeat(np.arange(self.num_users), self.num_items)
        item_ids = np.tile(np.arange(self.num_items), self.num_users)
        pairs = np.vstack([user_ids, item_ids]).T
        predictions = self.predict(pairs)
        return pd.DataFrame({
            "UserID": user_ids,
            "ItemID": item_ids,
            "Prediction": predictions.flatten()
        }).pivot(index='UserID', columns='ItemID', values='Prediction')
    
class ClienteFedRecSys:

    def __init__(self, id, modelo_global, avaliacoes_locais):
        self.id = id
        self.modelo = modelo_global
        self.avaliacoes_locais = avaliacoes_locais
        self.modelo_loss = None
        self.modelo_loos_indv = None

        self.X_train = None
        self.y_train = None
        

    def adicionar_novas_avaliacoes(self, quantidade, aleatorio=False):
        def item_nao_avaliado(item_id, avaliacoes_locais, user_id):
            return all((user_id, item_id) != (user, item) for user, item, _ in avaliacoes_locais)


        novas_avaliacoes = []

        items_nao_avaliados = [item_id for item_id in range(self.modelo.num_items) if item_nao_avaliado(item_id, self.avaliacoes_locais, self.id)]
        
        print(f"adicionar_novas_avaliacoes :: items_nao_avaliados :: UserID {self.id}")
        print(items_nao_avaliados)

        random.shuffle(items_nao_avaliados)
        
        for _ in range(quantidade):
            if not items_nao_avaliados:
                break  # Se não houver mais itens não avaliados, interromper
            item_id = items_nao_avaliados.pop()
            rating = random.randint(1, 5) if aleatorio else (_ % 5) + 1
            avaliacao = (self.id, item_id, rating)
            novas_avaliacoes.append(avaliacao)

        self.avaliacoes_locais += novas_avaliacoes

        # Atualizar self.X_train e self.y_train com os novos dados adicionados de forma eficiente
        novos_X = np.array([[self.id, item[1]] for item in novas_avaliacoes])
        novos_y = np.array([item[2] for item in novas_avaliacoes])

        if self.X_train is None:
            self.X_train = novos_X
            self.y_train = novos_y
        else:
            self.X_train = np.concatenate((self.X_train, novos_X))
            self.y_train = np.concatenate((self.y_train, novos_y))





    def treinar_modelo(self, learning_rate=0.02, epochs=2, batch_size=32, verbose=1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.modelo.compile(optimizer=optimizer, loss='mean_squared_error')
        self.modelo.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.modelo.compile(optimizer=optimizer, loss='mean_squared_error')
        history = self.modelo.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
        # Armazenar a perda do modelo após o treinamento
        self.modelo_loss = history.history['loss'][-1]

        # Calcular e armazenar a perda individual referente ao usuário
        loss_usuario = 0
        for usuario, item_id, rating in self.avaliacoes_locais:
            user_input = np.array([[usuario, item_id]])
            prediction = self.modelo.predict(user_input)[0]
            loss_usuario += (rating - prediction) ** 2
        
        num_avaliacoes = len(self.avaliacoes_locais)
        self.modelo_loss_indv = loss_usuario / num_avaliacoes

class ServidorFedRecSys:

    def __init__(self):
        self.modelo_global = None
        self.modelos_locais = []
        self.modelos_locais_loss = []
        self.modelos_locais_loss_indv = []
        self.numero_de_usuarios = None
        self.numero_de_itens = None
        self.avaliacoes_inicial = None
        self.avaliacoes_inicial_tensor = None
        self.avaliacoes_final_tensor = None # Este atributo foi adicionado apenas para comparar a injustiça nos métodos de agregação. Em um servidor real, não deveria existir.

        self.avaliacoes = None
        self.X_train = None
        self.y_train = None

    
    def iniciar_modelo(self, arquivo_excel):
        df = pd.read_excel(arquivo_excel, index_col=0)  # Especifica a primeira coluna como índice
        dados = df.fillna(0).values  # Preenche valores nulos com 0 e obtém os valores como array
        X, y = np.nonzero(dados)
        ratings = dados[X, y]
        self.numero_de_usuarios = df.shape[0]
        self.numero_de_itens = df.shape[1]
        self.avaliacoes_inicial = df
        avaliacoes_tuplas = list(zip(X, y, ratings))
        self.avaliacoes = avaliacoes_tuplas
        
        # Preparar os dados X_train e y_train
        self.X_train = np.array([[usuario, item] for usuario, item, _ in avaliacoes_tuplas])
        self.y_train = np.array([rating for _, _, rating in avaliacoes_tuplas])
        
        embedding_dim = 32
        self.modelo_global = MatrixFactorizationNN(self.numero_de_usuarios, self.numero_de_itens, embedding_dim)

        
    def treinar_modelo(self, learning_rate=0.02, epochs=2, batch_size=32, verbose=1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.modelo_global.compile(optimizer=optimizer, loss='mean_squared_error')
        self.modelo_global.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)


    def adicionar_avaliacoes_cliente(self, novas_avaliacoes):
        for tupla in novas_avaliacoes:
            if tupla not in self.avaliacoes:
                self.avaliacoes.append(tupla)


    def agregar_modelos_locais_ao_global_media_aritmetica_pesos(self, modelos_clientes):
        pesos_globais = self.modelo_global.get_weights()
        pesos_clientes = [cliente.get_weights() for cliente in modelos_clientes]

        novos_pesos = []
        for i, _ in enumerate(pesos_globais):
            pesos_parametro = [pesos[i] for pesos in pesos_clientes]
            media_parametro = np.mean(pesos_parametro, axis=0)
            novos_pesos.append(media_parametro)

        self.modelo_global.set_weights(novos_pesos)



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


def converter_tuplas_para_dataframe(tuplas, numero_de_usuarios, numero_de_itens):
    print("converter_tuplas_para_dataframe :: tuplas")
    print(tuplas)
    df = pd.DataFrame(columns=range(numero_de_itens), index=range(numero_de_usuarios))
    for tupla in tuplas:
        user_id, item_id, rating = tupla
        df.at[user_id, item_id] = rating
    return df

def iniciar_FedFairRecSys (dataset, G, rounds = 1, epochs=5, learning_rate=0.02, metodo_agregacao = 'ma'):

    print(f"\nMÉTODO DE AGREGAÇÃO :: {metodo_agregacao}")

    servidor = ServidorFedRecSys()
    servidor.iniciar_modelo(dataset)

    servidor.treinar_modelo(learning_rate=learning_rate, epochs=epochs, batch_size=32, verbose=1)
    
    print("servidor.avaliacoes_inicial")
    print(servidor.avaliacoes_inicial)

    print("servidor.numero_de_usuarios")
    print(servidor.numero_de_usuarios)

    print("servidor.numero_de_itens")
    print(servidor.numero_de_itens)

    print("servidor.redomendacoes")
    print(servidor.modelo_global.predict_all())

    print(f"INSTANCIANDO CLIENTES LOCAIS")
    clientes = []
    for i in range(servidor.numero_de_usuarios):
         # print(f"Cliente {i} :: Instanciando")
         cliente = ClienteFedRecSys(i, servidor.modelo_global, [avaliacao for avaliacao in servidor.avaliacoes if avaliacao[0] == i])
         clientes.append(cliente)

    print(f"TREINANDO CLIENTES LOCAIS")
    for round in range (rounds):
        print(f"\nRound: {round}")

        servidor.modelos_locais = []
        servidor.modelos_locais_loss = []
        servidor.modelos_locais_loss_indv = []
    #     servidor.avaliacoes_final_tensor = None
        
        for cliente in clientes:
            # print(f"Cliente {cliente.id} :: Adicionando Avaliações e Treinando")

            cliente.adicionar_novas_avaliacoes(quantidade=2, aleatorio=False)

    #         if cliente.id < 15:
    #             cliente.adicionar_novas_avaliacoes(10, False)
    #         else:
    #             cliente.adicionar_novas_avaliacoes(1, False)
            cliente.treinar_modelo(learning_rate=learning_rate, epochs=epochs, batch_size=32, verbose=1)

            print(f"cliente.modelo_loss {cliente.modelo_loss}")
            print(f"cliente.modelo_rindv {cliente.modelo_loss_indv}")

            servidor.modelos_locais.append(cliente.modelo)
            servidor.modelos_locais_loss.append(cliente.modelo_loss)
            servidor.modelos_locais_loss_indv.append(cliente.modelo_loss_indv)
            servidor.adicionar_avaliacoes_cliente(copy.deepcopy(cliente.avaliacoes_locais))
            
            print("cliente.avaliacoes_locais")
            print(cliente.avaliacoes_locais)

        if metodo_agregacao == 'ma':
            servidor.agregar_modelos_locais_ao_global_media_aritmetica_pesos(servidor.modelos_locais)
        elif metodo_agregacao == 'mp_loss':
            servidor.agregar_modelos_locais_ao_global_media_poderada_pesos_loss(servidor.modelos_locais, servidor.modelos_locais_loss)
        elif metodo_agregacao == 'mp_rindv':
            servidor.agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(servidor.modelos_locais, servidor.modelos_locais_loss_indv)

    # with torch.no_grad():
    #         recomendacoes_tensor = servidor.modelo_global(servidor.avaliacoes_final_tensor)

        avaliacoes_df = converter_tuplas_para_dataframe(servidor.avaliacoes, servidor.numero_de_usuarios, servidor.numero_de_itens)
        recomendacoes_df = servidor.modelo_global.predict_all()

        print("avaliacoes_df")
        print(avaliacoes_df)

        print("recomendacoes_df")
        print(recomendacoes_df)

            

    # # print("servidor.avaliacoes_final_tensor")
    # # print(servidor.avaliacoes_final_tensor)

    # # print("recomendacoes_final_01_ma_tensor")
    # # print(recomendacoes_final_01_ma_tensor)

    # print("=== MEDIDAS DE JUSTIÇA ===")

    # avaliacoes = servidor.avaliacoes_final_tensor.numpy()
    # avaliacoes_df = pd.DataFrame(avaliacoes)

    # recomendacoes_np = recomendacoes_tensor.numpy()
    # recomendacoes_df = pd.DataFrame(recomendacoes_np)

    # omega = (avaliacoes_df != 0)  

    # polarization = Polarization()
    # Rpol = polarization.evaluate(recomendacoes_df)

    # ilv = IndividualLossVariance(avaliacoes_df, omega, 1) #axis = 1 (0 rows e 1 columns)
    # Rindv = ilv.evaluate(recomendacoes_df)

    # glv = GroupLossVariance(avaliacoes_df, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    # Rgrp = glv.evaluate(recomendacoes_df)

    # rmse = RMSE(avaliacoes_df, omega)
    # result_rmse = rmse.evaluate(recomendacoes_df)

    # print(f"Polarization (Rpol) : {Rpol:.9f}")
    # print(f"Individual Loss Variance (Rindv) : {Rindv:.9f}")
    # print(f"Group Loss Variance (Rgrp) : {Rgrp:.9f}")
    # print(f'RMSE : {result_rmse:.9f}')

    # avaliacoes_df.to_excel(f"{dataset}-avaliacoes_df-{metodo_agregacao}.xlsx", index=False)
    # recomendacoes_df.to_excel(f"{dataset}-recomendacoes_df-{metodo_agregacao}.xlsx", index=False)



dataset='X-u5-i10_semindices.xlsx'
G = {1: list(range(0, 2)), 2: list(range(2, 5))}

# dataset='X.xlsx'
# G = {1: list(range(0, 15)), 2: list(range(15, 300))}

rounds=1
epochs=2
learning_rate=0.02
print(f"\nFedFairRecSys")
iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, metodo_agregacao='ma')
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, metodo_agregacao='mp_loss')
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, metodo_agregacao='mp_rindv')



    