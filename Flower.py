import numpy as np
import pandas as pd
import flwr as fl
from tensorflow.keras import layers, Model

# Leitura do arquivo Excel no formato de uma matriz indexada
df = pd.read_excel("X-u5-i10.xlsx", index_col=0)

user_ids = list(df.index)
item_ids = list(df.columns)

train_data = []
for user_id in user_ids:
    for item_id in item_ids:
        rating = df.loc[user_id, item_id]
        if not pd.isnull(rating):
            train_data.append((user_id, item_id, rating))

user_ids, item_ids, ratings = zip(*train_data)
user_ids = pd.Categorical(user_ids).codes
item_ids = pd.Categorical(item_ids).codes
ratings = np.array(ratings)

class RecommenderNet(Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(RecommenderNet, self).__init__()
        self.embedding_size = embedding_size
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = layers.Embedding(num_users, embedding_size, name='user_embedding')
        self.item_embedding = layers.Embedding(num_items, embedding_size, name='item_embedding')
        self.dot = layers.Dot(axes=1)

    def call(self, inputs):
        user_input, item_input = inputs
        user_vec = self.user_embedding(user_input)
        user_vec = layers.Flatten()(user_vec)
        item_vec = self.item_embedding(item_input)
        item_vec = layers.Flatten()(item_vec)

        # Modificação para operação do Dot diretamente
        dot_product = self.dot([user_vec, item_vec])

        return dot_product


class RecommenderClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val, user_id, num_items):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.user_id = user_id
        self.num_items = num_items
        self.local_ratings = set(zip(x_train, y_train))
    
    def get_parameters(self):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, validation_data=(self.x_val, self.y_val))
        return self.get_parameters(), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        loss = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"loss": loss}
    
    def generate_new_ratings(self, num_new_ratings):
        new_ratings = set()
        items_rated = set(item[0] for item in self.local_ratings)
        
        while len(new_ratings) < num_new_ratings:
            item_id = np.random.randint(1, self.num_items + 1)  
            if item_id not in items_rated:
                rating = np.random.randint(1, 6)  
                new_ratings.add((self.user_id, item_id, rating))
                items_rated.add(item_id)
        
        return new_ratings

class RecommenderServer(fl.server.Server):
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = []
        self.global_model = None
        self._client_manager = fl.server.SimpleClientManager()

    def client_manager(self):
        return self._client_manager

    def get_parameters(self):
        if self.global_model is None:
            self.global_model = RecommenderNet(len(user_ids), len(item_ids), 50)
        return self.global_model.get_weights()

    def fit(self, parameters):
        aggregated_weights = parameters
        for client in self.clients:
            aggregated_weights = self.aggregate_weights(aggregated_weights, client.get_parameters(), len(client.x_train))

        if self.global_model is None:
            self.global_model = RecommenderNet(len(user_ids), len(item_ids), 50)
        self.global_model.set_weights(aggregated_weights)
        return self.global_model.get_weights()

    def aggregate_weights(self, weights_old, weights_new, num_samples):
        new_weights = [(w1 * num_samples + w2 * num_samples) / (num_samples + num_samples) for w1, w2 in zip(weights_old, weights_new)]
        return new_weights

    def add_client(self, client):
        self.clients.append(client)

# Crie o servidor e adicione os clientes
server = RecommenderServer(num_clients=5)

# Adicione os clientes ao servidor
for client_id in range(5):
    model = RecommenderNet(len(user_ids), len(item_ids), 50)
    client = RecommenderClient(model, user_ids, item_ids, ratings, np.random.choice(ratings, len(ratings)), client_id + 1, len(item_ids))
    server.add_client(client)

# Simule treinamento federado com 2 rounds sem FedAVG
for round_num in range(2):
    print(f"\nRound {round_num + 1}")

    for client in server.clients:
        new_ratings = client.generate_new_ratings(num_new_ratings=1)
        print(f"Cliente {client.user_id} gerou novas avaliações: {new_ratings}")

    global_weights = server.get_parameters()

    for client in server.clients:
        client.set_parameters(global_weights)
        client.fit(global_weights, config=None)

    aggregated_weights = server.fit(global_weights)
    server.global_model.set_weights(aggregated_weights)

# Exibir dados de precisão do modelo global final
loss, acc, metrics = server.global_model.evaluate(...)
print(f"\nPrecisão do modelo global final - Loss: {loss}, Accuracy: {acc}, Outras métricas: {metrics}")
