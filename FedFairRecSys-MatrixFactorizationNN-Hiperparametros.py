import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
import tensorflow as tf

class EstimadorTensorFlow(BaseEstimator):
    def __init__(self, modelo):
        self.modelo = modelo
    
    def fit(self, X, y):
        self.modelo.fit(X, y)
        return self
    
    def predict(self, X):
        return self.modelo.predict(X)

class MatrixFactorizationNN(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorizationNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim,
                                                        embeddings_initializer='he_normal',
                                                        embeddings_regularizer=tf.keras.regularizers.l2(0.001))
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim,
                                                        embeddings_initializer='he_normal',
                                                        embeddings_regularizer=tf.keras.regularizers.l2(0.001))

    def call(self, inputs):
        user_id = inputs[:, 0]
        item_id = inputs[:, 1]
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        interaction = tf.multiply(user_vec, item_vec)
        score = tf.reduce_sum(interaction, axis=1)
        predicted_ratings = 1 + 4.0 * tf.sigmoid(score)
        return predicted_ratings


# Leitura dos dados do arquivo excel
arquivo_excel='X.xlsx'
df = pd.read_excel(arquivo_excel, index_col=0)
dados = df.fillna(0).values
X, y = np.nonzero(dados)
ratings = dados[X, y]

# Número de usuários e itens
numero_de_usuarios = df.shape[0]
numero_de_itens = df.shape[1]
avaliacoes_tuplas = list(zip(X, y, ratings))

X_train = np.array([[usuario, item] for usuario, item, _ in avaliacoes_tuplas])
y_train = np.array([rating for _, _, rating in avaliacoes_tuplas])

embedding_dim = 32
modelo = MatrixFactorizationNN(numero_de_usuarios, numero_de_itens, embedding_dim)

# Definição dos parâmetros a serem testados
# Definição dos parâmetros a serem testados (excluindo batch_size)
parametros_grid = {
    'epochs': [50, 100, 150],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Inicializar o GridSearchCV com o novo estimador adaptado
modelo_wrapper = EstimadorTensorFlow(modelo)
modelo_grid = GridSearchCV(modelo_wrapper, parametros_grid, scoring='neg_mean_squared_error', cv=3)

# Treinar o modelo GridSearchCV
modelo_grid.fit(X_train, y_train)

# Obter os melhores parâmetros
melhores_parametros = modelo_grid.best_params_
melhor_modelo = modelo_grid.best_estimator_

print("Melhores parâmetros encontrados:")
print(melhores_parametros)
