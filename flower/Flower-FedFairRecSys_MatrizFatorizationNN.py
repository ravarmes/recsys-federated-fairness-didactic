import flwr as fl
from flwr.server.strategy import FedAvg  # Certifique-se de importar FedAvg corretamente

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import copy
from sklearn.metrics import mean_squared_error
import time  # Para dar tempo para o servidor iniciar
from threading import Thread  # Para executar servidor e clientes em paralelo

# Classe do modelo de recomendação
class MatrixFactorizationNN(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

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


# Classe do cliente federado
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cliente_id, modelo, avaliacoes_locais, learning_rate=0.02, epochs=10, batch_size=32):
        self.cliente_id = cliente_id
        self.modelo = modelo
        self.avaliacoes_locais = avaliacoes_locais
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.carregar_dados()
    
    def get_parameters(self):  # Sem o argumento `config`
        return self.modelo.get_weights()  # Retorna os pesos do modelo
    
    def set_parameters(self, parameters):
        self.modelo.set_weights(parameters)

    def train(self):
        self.modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
            loss='mean_squared_error'
        )
        self.modelo.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        loss = self.modelo.evaluate(self.X_train, self.y_train, verbose=0)
        return loss, len(self.X_train), {"loss": loss}

    def carregar_dados(self):
        # Extrair IDs de clientes e itens
        novos_X = np.array([[self.cliente_id, item[1]] for item in self.avaliacoes_locais])
        # Extrair avaliações associadas
        novos_y = np.array([item[2] for item in self.avaliacoes_locais])
        
        # Se os arrays de treinamento estão vazios, inicialize-os
        if self.X_train is None:
            self.X_train = novos_X
            self.y_train = novos_y
        # Se já estiverem inicializados, concatene com novos dados
        else:
            self.X_train = np.concatenate((self.X_train, novos_X))
            self.y_train = np.concatenate((self.y_train, novos_y))


# Classe do servidor federado
class ServidorFedRecSys:
    def __init__(self):
        self.modelo_global = None
        self.modelos_locais = []
        self.modelos_locais_loss = []
        self.modelos_locais_loss_indv = []
        self.numero_de_usuarios = None
        self.numero_de_itens = None
        self.avaliacoes = None

    def iniciar_modelo(self, arquivo_excel):
        df = pd.read_excel(arquivo_excel, index_col=0)  # Ler dados do Excel
        dados = df.fillna(0).values  # Substituir valores nulos por zero
        X, y = np.nonzero(dados)  # Obter índices não nulos
        ratings = dados[X, y]  # Obter avaliações
        self.numero_de_usuarios = df.shape[0]
        self.numero_de_itens = df.shape[1]
        self.avaliacoes = list(zip(X, y, ratings))
        embedding_dim = 32  # Dimensão dos embeddings
        self.modelo_global = MatrixFactorizationNN(self.numero_de_usuarios, self.numero_de_itens, embedding_dim)

    def treinar_modelo(self, learning_rate=0.02, epochs=5, batch_size=32, verbose=1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Otimizador
        self.modelo_global.compile(optimizer=optimizer, loss='mean_squared_error')  # Compilar modelo
        self.modelo_global.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)  # Treinar modelo

    def agregar_modelos_locais_ao_global_media_aritmetica_pesos(self, modelos_clientes):
        pesos_globais = self.modelo_global.get_weights()  # Obter pesos globais
        pesos_clientes = [cliente.get_weights() for cliente in modelos_clientes]  # Pesos dos clientes
        novos_pesos = []
        for i, _ in enumerate(pesos_globais):
            pesos_parametro = [pesos[i] for pesos in pesos_clientes]
            media_parametro = np.mean(pesos_parametro, axis=0)
            novos_pesos.append(media_parametro)
        self.modelo_global.set_weights(novos_pesos)  # Atualizar pesos do modelo global

    def agregar_modelos_locais_ao_global_media_poderada_pesos_loss(self, modelos_clientes, modelos_clientes_loss):
        total_perdas = sum(modelos_clientes_loss)  # Soma das perdas
        pesos = [perda / total_perdas for perda in modelos_clientes_loss]
        pesos_globais = self.modelo_global.get_weights()
        pesos_clientes = [cliente.get_weights() for cliente in modelos_clientes]
        novos_pesos = []
        for i, _ in enumerate(pesos_globais):
            pesos_parametro = np.dot(pesos, [peso[i] for peso in pesos_clientes])
            novos_pesos.append(pesos_parametro)
        self.modelo_global.set_weights(novos_pesos)  # Atualizar pesos do modelo global

    def agregar_modelos_locais_ao_global_media_poderada_pesos_rindv(self, modelos_clientes, modelos_clientes_loss_indv):
        total_perdas = sum(modelos_clientes_loss_indv)
        pesos = [perda / total_perdas for perda in modelos_clientes_loss_indv]
        pesos_globais = self.modelo_global.get_weights()
        pesos_clientes = [cliente.get_weights() for cliente in modelos_clientes]
        novos_pesos = []
        for i, _ in enumerate(pesos_globais):
            pesos_parametro = np.dot(pesos, [peso[i] for peso in pesos_clientes])
            novos_pesos.append(pesos_parametro)

        # Definir os novos pesos no modelo global
        self.modelo_global.set_weights(novos_pesos) 


# Função para conectar clientes ao servidor federado
def conectar_clientes_flower(cliente_id, avaliacoes_locais, modelo):
    cliente = FlowerClient(cliente_id, modelo, avaliacoes_locais)

    # Conectar ao servidor federado
    fl.client.start_client(
        server_address="127.0.0.1:8080",  # Novo método recomendado
        client=cliente.to_client(),  # Uso de `.to_client()`
    )


# Função for iniciar o servidor federado
def iniciar_servidor_flower():
    # Estratégia de agregação para o servidor federado
    strategy = FedAvg(
        min_fit_clients=10,  # Número mínimo de clientes para treinamento
        min_available_clients=15,  # Número mínimo de clientes disponíveis
    )

    # Iniciar o servidor federado
    fl.server.start_server(
        server_address="127.0.0.1:8080",  # Endereço do servidor federado
        strategy=strategy,  # Passar a estratégia configurada
    )



# Função para iniciar o sistema federado e conectar os clientes
def iniciar_sistema_federado():
    # Iniciar servidor federado
    servidor_thread = Thread(target=iniciar_servidor_flower)
    servidor_thread.start()

    # Aguardar alguns segundos para que o servidor inicie
    time.sleep(8)

    # Conectar dois clientes ao servidor federado
    cliente1_thread = Thread(
        target=conectar_clientes_flower,
        args=(1, [], MatrixFactorizationNN(100, 100, 32))
    )
    cliente1_thread.start()

    cliente2_thread = Thread(
        target=conectar_clientes_flower,
        args=(2, [], MatrixFactorizationNN(100, 100, 32))
    )
    cliente2_thread.start()

    # Esperar o término do servidor e dos clientes
    servidor_thread.join()
    cliente1_thread.join()
    cliente2_thread.join()


# Executar o sistema federado com um servidor e dois clientes
iniciar_sistema_federado()
