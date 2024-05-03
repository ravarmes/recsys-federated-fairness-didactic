# import os
# import sys

# Redirecionando stdout e stderr para /dev/null
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

# Definir o nível de log para 'error' no TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import copy
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance
from AlgorithmImpartiality import AlgorithmImpartiality

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
        df = pd.DataFrame({
            "UserID": user_ids,
            "ItemID": item_ids,
            "Prediction": predictions.flatten()
        })
        df['Prediction'] = df['Prediction'].astype(float)

        df_no_names = df.pivot(index='UserID', columns='ItemID', values='Prediction')
        df_no_names.columns.name = None
        df_no_names.index.name = None
    
        return df_no_names

    
class ClienteFedRecSys:

    def __init__(self, id, modelo_global, avaliacoes_locais):
        self.id = id
        self.modelo = modelo_global
        self.avaliacoes_locais = avaliacoes_locais
        self.modelo_loss = None
        self.modelo_loss_indv = None
        self.modelo_mean_indv = None

        self.X_train = None
        self.y_train = None

        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.modelo.compile(optimizer=optimizer, loss='mean_squared_error')
        

    def adicionar_novas_avaliacoes(self, quantidade, aleatorio=False):
        def item_nao_avaliado(item_id, avaliacoes_locais, user_id):
            return all((user_id, item_id) != (user, item) for user, item, _ in avaliacoes_locais)

        novas_avaliacoes = []
        items_nao_avaliados = [item_id for item_id in range(self.modelo.num_items) if item_nao_avaliado(item_id, self.avaliacoes_locais, self.id)]
        # print(f"adicionar_novas_avaliacoes :: items_nao_avaliados :: UserID {self.id}")
        # print(items_nao_avaliados)
        random.shuffle(items_nao_avaliados)
        
        for _ in range(quantidade):
            if not items_nao_avaliados:
                break  # Se não houver mais itens não avaliados, interromper
            item_id = items_nao_avaliados.pop()
            rating = random.randint(1, 5) if aleatorio else (_ % 5) + 1
            avaliacao = (self.id, item_id, float(rating))
            novas_avaliacoes.append(avaliacao)

        self.avaliacoes_locais += novas_avaliacoes

        novos_X = np.array([[self.id, item[1]] for item in novas_avaliacoes])
        novos_y = np.array([item[2] for item in novas_avaliacoes])

        if self.X_train is None:
            self.X_train = novos_X
            self.y_train = novos_y
        else:
            self.X_train = np.concatenate((self.X_train, novos_X))
            self.y_train = np.concatenate((self.y_train, novos_y))



    def treinar_modelo(self, epochs=2, batch_size=32, verbose=1):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.modelo.compile(optimizer=optimizer, loss='mean_squared_error')
        history = self.modelo.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
        # Armazenar a perda do modelo após o treinamento
        self.modelo_loss = history.history['loss'][-1] 

        # Criar um array para armazenar os inputs
        user_inputs = np.array([[usuario, item_id] for usuario, item_id, _ in self.avaliacoes_locais])

        # Obter as previsões para todos os inputs de uma vez
        predictions = self.modelo.predict(user_inputs)

        # Calcular a perda individual para o usuário específico
        loss_usuario_especifico = 0
        mean_usuario_especifico = 0
        for i, (_, _, rating) in enumerate(self.avaliacoes_locais):
            prediction = predictions[i]
            loss_usuario_especifico += (rating - prediction) ** 2
            mean_usuario_especifico += (rating - prediction)

        # Calcular o número total de avaliações do usuário específico
        num_avaliacoes_usuario_especifico = len(self.avaliacoes_locais)

        # Calcular a perda individual média para o usuário específico
        modelo_loss_indv_usuario_especifico = loss_usuario_especifico / num_avaliacoes_usuario_especifico
        self.modelo_loss_indv = modelo_loss_indv_usuario_especifico

        # Calcular a diferença média individual para o usuário específico
        modelo_mean_indv_usuario_especifico = mean_usuario_especifico / num_avaliacoes_usuario_especifico
        self.modelo_mean_indv = modelo_mean_indv_usuario_especifico

        # print("\ntreinar_modelo :: self.avaliacoes_locais")
        # print(self.avaliacoes_locais)

        # print("\ntreinar_modelo :: predictions")
        # print(predictions)

        # print("\ntreinar_modelo :: self.modelo.predict_all()")
        # print(self.modelo.predict_all())

        # print("\ntreinar_modelo :: self.modelo_loss_indv")
        # print(self.modelo_loss_indv)

        # print("\ntreinar_modelo :: self.modelo_mean_indv")
        # print(self.modelo_mean_indv)



class ServidorFedRecSys:
    def __init__(self):
        self.modelo_global = None

        self.modelos_locais = []
        self.modelos_locais_loss = []
        self.modelos_locais_loss_indv = []
        self.modelos_locais_mean_indv = []

        self.numero_de_usuarios = None
        self.numero_de_itens = None

        self.avaliacoes = None # Este atributo foi adicionado apenas para comparar a injustiça nos métodos de agregação. Em um servidor real, não deveria existir.
        self.X_train = None
        self.y_train = None

    
    def iniciar_modelo(self, arquivo_excel, learning_rate, embedding_dim):
        df = pd.read_excel(arquivo_excel, index_col=0)  # Especifica a primeira coluna como índice
        dados = df.fillna(0).values  # Preenche valores nulos com 0 e obtém os valores como array
        X, y = np.nonzero(dados)
        ratings = dados[X, y]
        self.numero_de_usuarios = df.shape[0]
        self.numero_de_itens = df.shape[1]
        avaliacoes_tuplas = list(zip(X, y, ratings))
        self.avaliacoes = avaliacoes_tuplas
        
        # Preparar os dados X_train e y_train
        self.X_train = np.array([[usuario, item] for usuario, item, _ in avaliacoes_tuplas])
        self.y_train = np.array([rating for _, _, rating in avaliacoes_tuplas])
        
        self.modelo_global = MatrixFactorizationNN(self.numero_de_usuarios, self.numero_de_itens, embedding_dim)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.modelo_global.compile(optimizer=optimizer, loss='mean_squared_error')

        
    def treinar_modelo(self, epochs=2, batch_size=32, verbose=1):
        self.modelo_global.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)


    def adicionar_avaliacoes_cliente(self, avaliacoes):
        # print(f"avaliacoes: {avaliacoes}")
        novas_avaliacoes = []
        for tupla in avaliacoes:
            if tupla not in self.avaliacoes:
                self.avaliacoes.append(tupla)
                novas_avaliacoes.append(tupla)
        # print(f"adicionar_avaliacoes_cliente :: novas_avaliacoes: {novas_avaliacoes}")

        novos_X = np.array([[item[0], item[1]] for item in novas_avaliacoes])
        novos_y = np.array([item[2] for item in novas_avaliacoes])

        # print("adicionar_avaliacoes_cliente :: novos_X")
        # print(novos_X)

        if self.X_train is None:
            self.X_train = novos_X
            self.y_train = novos_y
        else:
            self.X_train = np.concatenate((self.X_train, novos_X))
            self.y_train = np.concatenate((self.y_train, novos_y))

        # print("adicionar_avaliacoes_cliente :: self.X_train")
        # print(self.X_train)


    def agregar_modelos_locais_ao_global_media_aritmetica_pesos(self):
        pesos_globais = self.modelo_global.get_weights()
        pesos_clientes = [cliente.get_weights() for cliente in self.modelos_locais]
        novos_pesos = []
        for i, _ in enumerate(pesos_globais):
            pesos_parametro = [pesos[i] for pesos in pesos_clientes]
            media_parametro = np.mean(pesos_parametro, axis=0)
            novos_pesos.append(media_parametro)
        self.modelo_global.set_weights(novos_pesos)


    # def agregar_modelos_locais_ao_global_media_poderada_pesos_loss(self):
    #     total_perdas = sum(self.modelos_locais_loss)
    #     pesos = [perda / total_perdas for perda in self.modelos_locais_loss]
    #     pesos_globais = self.modelo_global.get_weights()
    #     pesos_clientes = [cliente.get_weights() for cliente in self.modelos_locais]
    #     novos_pesos = []
    #     for i, _ in enumerate(pesos_globais):
    #         pesos_parametro = [pesos[i] for pesos in pesos_clientes]
    #         media_parametro = np.mean(pesos_parametro, axis=0)
    #         novos_pesos.append(media_parametro)
    #     self.modelo_global.set_weights(novos_pesos)

    
    # def agregar_modelos_locais_ao_global_media_poderada_pesos_loss_indv(self):
    #     total_perdas = sum(self.modelos_locais_loss_indv)
    #     pesos = [perda / total_perdas for perda in self.modelos_locais_loss_indv]
    #     pesos_globais = self.modelo_global.get_weights()
    #     pesos_clientes = [cliente.get_weights() for cliente in self.modelos_locais]
    #     novos_pesos = []
    #     for i, _ in enumerate(pesos_globais):
    #         pesos_parametro = [pesos[i] for pesos in pesos_clientes]
    #         media_parametro = np.mean(pesos_parametro, axis=0)
    #         novos_pesos.append(media_parametro)
    #     self.modelo_global.set_weights(novos_pesos)

    def agregar_modelos_locais_ao_global_media_ponderada_pesos_loss(self):
        total_perdas = sum(self.modelos_locais_loss)
        pesos_normalizados = np.array([1 / (perda / total_perdas) for perda in self.modelos_locais_loss])
        pesos_normalizados /= pesos_normalizados.sum()
        pesos_clientes = [cliente.get_weights() for cliente in self.modelos_locais]
        novos_pesos = [
            np.average([cliente_pesos[idx] for cliente_pesos in pesos_clientes], axis=0, weights=pesos_normalizados)
            for idx in range(len(self.modelo_global.get_weights()))
        ]
        self.modelo_global.set_weights(novos_pesos)


    def agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv(self):
        total_perdas = sum(self.modelos_locais_loss_indv)
        pesos_normalizados = np.array([1 / (perda / total_perdas) for perda in self.modelos_locais_loss_indv])
        pesos_normalizados /= pesos_normalizados.sum()

        print("agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv :: pesos_normalizados")
        print(pesos_normalizados)

        pesos_clientes = [cliente.get_weights() for cliente in self.modelos_locais]
        novos_pesos = [
            np.average([cliente_pesos[idx] for cliente_pesos in pesos_clientes], axis=0, weights=pesos_normalizados)
            for idx in range(len(self.modelo_global.get_weights()))
        ]
        self.modelo_global.set_weights(novos_pesos)


    # def agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv2(self):
    #     # Passo 1: Calcular a média dos erros individuais
    #     media_global = np.mean(self.modelos_locais_loss_indv)
        
    #     # Passo 2: Calcular as diferenças em relação à média
    #     diferencas = np.array(self.modelos_locais_loss_indv) - media_global
        
    #     # Passo 3: Normalizar as diferenças para uma escala de 0 a 1
    #     min_diferenca = np.min(diferencas)
    #     max_diferenca = np.max(diferencas)
        
    #     if max_diferenca - min_diferenca == 0:  # Evitar divisão por zero
    #         diferencas_normalizadas = np.zeros_like(diferencas)
    #     else:
    #         diferencas_normalizadas = (diferencas - min_diferenca) / (max_diferenca - min_diferenca)
        
    #     # Passo 4: Calcular pesos invertidos e normalizar
    #     pesos_invertidos = 1 - diferencas_normalizadas
    #     pesos_normalizados = pesos_invertidos / np.sum(pesos_invertidos)
        
    #     # Passo 5: Obter os pesos dos modelos locais e garantir consistência
    #     pesos_modelos_locais = [np.array(modelo.get_weights()) for modelo in self.modelos_locais]
        
    #     # Identificar a forma do maior modelo local para inicializar 'pesos_agg'
    #     max_shape = max([peso.shape for peso in pesos_modelos_locais], key=lambda x: np.prod(x))
    #     pesos_agg = np.zeros(max_shape)  # Inicializar com zeros
        
    #     # Passo 6: Aplicar a média ponderada para calcular pesos do modelo global
    #     for idx, peso in enumerate(pesos_modelos_locais):
    #         # Verificar se as formas são compatíveis antes de somar
    #         if pesos_agg.shape == peso.shape:
    #             pesos_agg += pesos_normalizados[idx] * peso
    #         else:
    #             raise ValueError("Incompatibilidade de formas entre modelos locais.")
        
    #     # Definir os pesos do modelo global
    #     self.modelo_global.set_weights(pesos_agg.tolist())  # Converter para lista antes de definir

    def agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv2(self):
        # Passo 1: Calcular a média dos erros individuais
        media_global = np.mean(self.modelos_locais_loss_indv)
        
        # Passo 2: Calcular as diferenças em relação à média
        diferencas = np.array(self.modelos_locais_loss_indv) - media_global
        
        # Passo 3: Normalizar as diferenças para uma escala de 0 a 1
        min_diferenca = np.min(diferencas)
        max_diferenca = np.max(diferencas)
        
        if max_diferenca - min_diferenca == 0:  # Evitar divisão por zero
            diferencas_normalizadas = np.zeros_like(diferencas)
        else:
            diferencas_normalizadas = (diferencas - min_diferenca) / (max_diferenca - min_diferenca)
        
        # Passo 4: Calcular pesos invertidos e normalizar
        # pesos_invertidos = 1 - diferencas_normalizadas
        pesos_normalizados = diferencas_normalizadas / np.sum(diferencas_normalizadas)

        print("agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv2 :: pesos_normalizados")
        print(pesos_normalizados)
        
        pesos_clientes = [cliente.get_weights() for cliente in self.modelos_locais]
        novos_pesos = [
            np.average([cliente_pesos[idx] for cliente_pesos in pesos_clientes], axis=0, weights=pesos_normalizados)
            for idx in range(len(self.modelo_global.get_weights()))
        ]
        self.modelo_global.set_weights(novos_pesos)


    # Este método está considerando apenas as recomendações (ou seja, os modelos locais enviados pelos clientes)
    def aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(self, G):
        recomendacoes = self.modelo_global.predict_all()
        omega = ~recomendacoes.isnull() 

        algorithmImpartiality = AlgorithmImpartiality(recomendacoes, omega, 1)
        list_X_est = algorithmImpartiality.evaluate_federated(recomendacoes, self.modelos_locais_mean_indv, self.modelos_locais_loss_indv, 10) # calculates a list of h estimated matrices => h = 5

        ilv = IndividualLossVariance(recomendacoes, omega, 1)
        list_losses = []
        for X_est in list_X_est:
            losses = ilv.get_losses(X_est)
            list_losses.append(losses)

        Z = AlgorithmImpartiality.losses_to_Z(list_losses)
        list_Zs = AlgorithmImpartiality.matrices_Zs(Z, G)
        recomendacoes_fairness = AlgorithmImpartiality.make_matrix_X_gurobi(list_X_est, G, list_Zs) 
        
        # Preparando o dataframe recomendacoes_fairness para dados no formato de tupla e treinar o modelo global

        dados = recomendacoes_fairness.fillna(0).values  # Preenche valores nulos com 0 e obtém os valores como array
        X, y = np.nonzero(dados)
        ratings = dados[X, y]
        self.numero_de_usuarios = recomendacoes_fairness.shape[0]
        self.numero_de_itens = recomendacoes_fairness.shape[1]
        avaliacoes_tuplas = list(zip(X, y, ratings))
        # self.avaliacoes = avaliacoes_tuplas
        
        # Preparar os dados X_train e y_train
        self.X_train = np.array([[usuario, item] for usuario, item, _ in avaliacoes_tuplas])
        self.y_train = np.array([rating for _, _, rating in avaliacoes_tuplas])
        
        self.modelo_global = MatrixFactorizationNN(self.numero_de_usuarios, self.numero_de_itens, embedding_dim)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.modelo_global.compile(optimizer=optimizer, loss='mean_squared_error')

    

    # Este método está considerando apenas as avaliações e as recomendações (ou seja, os modelos locais enviados pelos clientes + avaliações)
    # As avaliações não estarão disponíveis em um sistema federado. Mas, aqui fiz apenas para teste
    def aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global2(self, G):
        avaliacoes = converter_tuplas_para_dataframe(self.avaliacoes, self.numero_de_usuarios, self.numero_de_itens)
        recomendacoes = self.modelo_global.predict_all()
        omega = ~avaliacoes.isnull() 

        # # --------------------------------
        # print("\naplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global :: avaliacoes")
        # print(avaliacoes)
        # print(type(avaliacoes))

        # print("\naplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global :: recomendacoes")
        # print(recomendacoes)
        # print(type(recomendacoes))

        # # # Encontrar os valores não numéricos
        # # # Aplicar a função em todo o DataFrame para obter uma máscara booleana
        # # mask = recomendacoes.applymap(is_numeric)
        # # # Encontrar células não numéricas usando a máscara
        # # non_numeric_values = recomendacoes[~mask]
        # # print("Valores não numéricos:")
        # # print(non_numeric_values)

        # # # Converter todo o DataFrame para float, com coerção de erros
        # # recomendacoes = recomendacoes.apply(pd.to_numeric, errors='coerce')
        # # # Substituir NaN por um valor padrão (0) em todo o DataFrame
        # # recomendacoes.fillna(0, inplace=True)

        # # print("Verificando valores não numéricos")
        # # for column in recomendacoes.columns:
        # #     is_numeric = pd.to_numeric(recomendacoes[column], errors='coerce').notna()
        # #     if not is_numeric.all():
        # #         print(f"Valores não numéricos encontrados na coluna {column}:")
        # #         print(recomendacoes[column][~is_numeric])

        # recomendacoes = recomendacoes.apply(pd.to_numeric, errors='coerce')
        # recomendacoes.fillna(0, inplace=True)
        # recomendacoes.to_excel(f"teste.xlsx", index=False)


        # # -------------------------------------------

        ilv = IndividualLossVariance(avaliacoes, omega, 1)
        algorithmImpartiality = AlgorithmImpartiality(avaliacoes, omega, 1)
        list_X_est = algorithmImpartiality.evaluate(recomendacoes, 10) # calculates a list of h estimated matrices => h = 5

        list_losses = []
        for X_est in list_X_est:
            losses = ilv.get_losses(X_est)
            list_losses.append(losses)

        Z = AlgorithmImpartiality.losses_to_Z(list_losses)
        list_Zs = AlgorithmImpartiality.matrices_Zs(Z, G)
        recomendacoes_fairness = AlgorithmImpartiality.make_matrix_X_gurobi(list_X_est, G, list_Zs) 

        return recomendacoes_fairness
    

def converter_tuplas_para_dataframe(tuplas, numero_de_usuarios, numero_de_itens):
    df = pd.DataFrame(columns=range(numero_de_itens), index=range(numero_de_usuarios))
    for tupla in tuplas:
        user_id, item_id, rating = tupla
        df.at[user_id, item_id] = rating
    return df

# Função para testar se um valor pode ser convertido para float
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def iniciar_FedFairRecSys (dataset, G, rounds = 1, epochs=5, learning_rate=0.02, embedding_dim=16, metodo_agregacao = 'ma'):

    print("\n----------------------------------------------------")
    print(f"\nMÉTODO DE AGREGAÇÃO :: {metodo_agregacao}")

    servidor = ServidorFedRecSys()
    print("\nSERVIDOR INICIANDO MODELO")
    servidor.iniciar_modelo(dataset, learning_rate, embedding_dim)
    print("\nSERVIDOR TREINANDO O MODELO")
    servidor.treinar_modelo(epochs, batch_size=32, verbose=1)
    
    # print("servidor.avaliacoes_inicial")
    # print(servidor.avaliacoes_inicial)

    # print("servidor.numero_de_usuarios")
    # print(servidor.numero_de_usuarios)

    # print("servidor.numero_de_itens")
    # print(servidor.numero_de_itens)

    # print("servidor.redomendacoes")
    # print(servidor.modelo_global.predict_all())

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
        servidor.modelos_locais_mean_indv = []
        
        if metodo_agregacao != "nao_federado":
        
            for cliente in clientes:
                print(f"Cliente {cliente.id} :: Adicionando Avaliações e Treinando")
                # print("cliente.adicionar_novas_avaliacoes")
                
                # cliente.adicionar_novas_avaliacoes(quantidade=2, aleatorio=False)
                cliente.adicionar_novas_avaliacoes(20, False) if cliente.id < 15 else cliente.adicionar_novas_avaliacoes(2, False)

                # print("cliente.treinar_modelo")
                cliente.treinar_modelo(epochs, batch_size=32, verbose=1)

                # print(f"cliente.modelo_loss {cliente.modelo_loss}")
                # print(f"cliente.modelo_rindv {cliente.modelo_loss_indv}")

                # print("servidor.adicionar_avaliacoes_cliente")
                servidor.modelos_locais.append(cliente.modelo)
                servidor.modelos_locais_loss.append(cliente.modelo_loss)
                servidor.modelos_locais_loss_indv.append(cliente.modelo_loss_indv)
                servidor.modelos_locais_mean_indv.append(cliente.modelo_mean_indv)
                servidor.adicionar_avaliacoes_cliente(copy.deepcopy(cliente.avaliacoes_locais))
                
            # print("servidor.agregar_modelos_locais_ao_global")
            if metodo_agregacao == 'ma':
                servidor.agregar_modelos_locais_ao_global_media_aritmetica_pesos()
            elif metodo_agregacao == 'mp_loss':
                servidor.agregar_modelos_locais_ao_global_media_ponderada_pesos_loss()
            elif metodo_agregacao == 'mp_loss_indv':
                servidor.agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv()
            elif metodo_agregacao == 'mp_loss_indv2':
                servidor.agregar_modelos_locais_ao_global_media_ponderada_pesos_loss_indv2()
            elif metodo_agregacao == 'ma_fair':
                servidor.agregar_modelos_locais_ao_global_media_aritmetica_pesos()
                servidor.aplicar_algoritmo_imparcialidade_na_agregacao_ao_modelo_global(G)
                servidor.treinar_modelo(epochs)

        elif metodo_agregacao == 'nao_federado':
            for cliente in clientes:
                print(f"Cliente {cliente.id} :: Adicionando Avaliações")
                # print("cliente.adicionar_novas_avaliacoes")
                
                # cliente.adicionar_novas_avaliacoes(quantidade=2, aleatorio=False)
                cliente.adicionar_novas_avaliacoes(20, False) if cliente.id < 15 else cliente.adicionar_novas_avaliacoes(2, False)
                #print("servidor.adicionar_avaliacoes_cliente")
                servidor.adicionar_avaliacoes_cliente(copy.deepcopy(cliente.avaliacoes_locais))

            # print("servidor.treinar_modelo")
            servidor.treinar_modelo(epochs) # Considerando as novas avaliações dos clientes locais

    avaliacoes_df = converter_tuplas_para_dataframe(servidor.avaliacoes, servidor.numero_de_usuarios, servidor.numero_de_itens)
    recomendacoes_df = servidor.modelo_global.predict_all()
    omega = ~avaliacoes_df.isnull() 

    # print("\nservidor.X_train")
    # print(servidor.X_train)

    # print("\nservidor.y_train")
    # print(servidor.y_train)
    
    print("=== MEDIDAS DE JUSTIÇA ===")

    polarization = Polarization()
    Rpol = polarization.evaluate(recomendacoes_df)

    ilv = IndividualLossVariance(avaliacoes_df, omega, 1) #axis = 1 (0 rows e 1 columns)
    Rindv = ilv.evaluate(recomendacoes_df)

    glv = GroupLossVariance(avaliacoes_df, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    Rgrp = glv.evaluate(recomendacoes_df)

    rmse = RMSE(avaliacoes_df, omega)
    result_rmse = rmse.evaluate(recomendacoes_df)

    print(f"Método de Agregação {metodo_agregacao} ")
    print(f"Polarization (Rpol) : {Rpol:.9f}")
    print(f"Individual Loss Variance (Rindv) : {Rindv:.9f}")
    print(f"Group Loss Variance (Rgrp) : {Rgrp:.9f}")
    print(f'RMSE : {result_rmse:.9f}')


    # Defina o nome do arquivo onde deseja salvar as saídas
    output_file = f"resultados-{dataset}.txt"

    # Redirecione a saída dos prints para um arquivo txt
    with open(output_file, "a") as file:
        
        print(f"Round {rounds} - Epochs {epochs} -  Learning Rate {learning_rate}", file=file)
        print(f"Dataset {dataset} ", file=file)
        print(f"Método de Agregação {metodo_agregacao} ", file=file)
        print(f"Polarization (Rpol) : {Rpol:.9f}", file=file)
        print(f"Individual Loss Variance (Rindv) : {Rindv:.9f}", file=file)
        print(f"Group Loss Variance (Rgrp) : {Rgrp:.9f}", file=file)
        print(f'RMSE : {result_rmse:.9f}', file=file)
        print(f'\n', file=file)


    avaliacoes_df.to_excel(f"_xls/{dataset}-avaliacoes_df-{metodo_agregacao}.xlsx", index=False)
    recomendacoes_df.to_excel(f"_xls/{dataset}-recomendacoes_df-{metodo_agregacao}.xlsx", index=False)



# Agrupamento por Atividade
G_ACTIVITY = {1: list(range(0, 15)), 2: list(range(15, 300))}

# Agrupamento por Gênero
G_GENDER = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 96, 99, 100, 102, 103, 105, 107, 108, 109, 110, 111, 112, 115, 117, 118, 120, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 146, 147, 148, 149, 151, 152, 153, 154, 156, 159, 160, 161, 162, 164, 165, 166, 168, 169, 170, 172, 174, 175, 176, 177, 178, 181, 182, 183, 184, 186, 187, 188, 189, 191, 194, 196, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 218, 219, 220, 222, 223, 224, 226, 227, 229, 230, 231, 232, 233, 234, 237, 238, 239, 240, 245, 246, 247, 248, 249, 250, 251, 252, 255, 256, 257, 258, 259, 260, 261, 262, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 291, 292, 293, 294, 295, 296, 297, 298, 299], 2: [14, 25, 31, 34, 35, 42, 63, 73, 81, 91, 92, 97, 98, 101, 104, 106, 113, 114, 116, 119, 121, 122, 133, 144, 145, 150, 155, 157, 158, 163, 167, 171, 173, 179, 180, 185, 190, 192, 193, 195, 197, 199, 213, 214, 215, 216, 217, 221, 225, 228, 235, 236, 241, 242, 243, 244, 253, 254, 264, 290]}

# Agrupamento por Idade
G_AGE = {1: [14, 132, 194, 262, 273], 2: [8, 23, 26, 33, 48, 50, 61, 64, 70, 71, 76, 82, 86, 90, 92, 94, 96, 101, 107, 124, 126, 129, 134, 140, 149, 157, 158, 159, 163, 168, 171, 174, 175, 189, 191, 201, 207, 209, 215, 216, 222, 231, 237, 244, 246, 251, 255, 265, 270, 275, 282, 288, 290], 3: [3, 6, 7, 9, 10, 11, 15, 16, 21, 22, 24, 28, 29, 31, 32, 34, 35, 37, 39, 40, 41, 42, 43, 44, 45, 51, 53, 55, 56, 59, 60, 63, 65, 66, 69, 72, 73, 74, 75, 79, 80, 81, 85, 89, 93, 97, 102, 103, 104, 106, 108, 109, 110, 111, 116, 118, 119, 120, 122, 128, 130, 131, 133, 135, 136, 138, 139, 141, 142, 143, 145, 147, 151, 155, 161, 164, 169, 170, 173, 176, 179, 181, 183, 186, 187, 188, 190, 192, 193, 195, 196, 198, 200, 202, 203, 204, 205, 206, 211, 212, 213, 217, 219, 220, 223, 225, 226, 229, 230, 232, 233, 234, 236, 238, 240, 241, 249, 252, 253, 254, 258, 260, 261, 264, 267, 268, 269, 276, 277, 279, 280, 283, 285, 286, 287, 289, 291, 293, 294, 295, 296, 298], 4: [1, 2, 4, 5, 13, 17, 18, 25, 27, 36, 38, 49, 52, 57, 68, 77, 78, 84, 87, 88, 91, 95, 98, 99, 100, 105, 112, 117, 121, 127, 144, 146, 150, 152, 153, 156, 166, 172, 177, 182, 199, 208, 210, 214, 227, 228, 243, 245, 248, 250, 256, 263, 271, 272, 278, 292, 297, 299], 5: [19, 20, 30, 46, 47, 54, 58, 62, 67, 83, 113, 125, 137, 148, 160, 165, 167, 184, 197, 221, 235, 239, 242, 281], 6: [0, 114, 115, 123, 178, 180, 185, 224, 247, 257, 266, 274], 7: [12, 154, 162, 218, 259, 284]}

G = G_ACTIVITY

dataset='X.xlsx'

# rounds=3
# epochs=3 
# learning_rate=0.01
# embedding_dim = 16

# Melhores Hiperparâmetros
rounds=5 
epochs=10 
learning_rate=0.000174
embedding_dim = 16

# dataset='X-u5-i10_semindices.xlsx'
# G = {1: list(range(0, 2)), 2: list(range(2, 5))}

# rounds= 1
# epochs= 1
# learning_rate=0.1
# embedding_dim = 16


print(f"\nFedFairRecSys")
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, embedding_dim, metodo_agregacao='ma')
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, embedding_dim, metodo_agregacao='mp_loss')
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, embedding_dim, metodo_agregacao='mp_loss_indv')
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, embedding_dim, metodo_agregacao='mp_loss_indv2')
iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, embedding_dim, metodo_agregacao='ma_fair')
# iniciar_FedFairRecSys(dataset, G, rounds, epochs, learning_rate, embedding_dim, metodo_agregacao='nao_federado')



    