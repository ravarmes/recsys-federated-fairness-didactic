from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

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

class MatrixFactorizationHyperModel(HyperModel):
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items

    def build(self, hp):
        model = MatrixFactorizationNN(
            num_users=self.num_users, 
            num_items=self.num_items, 
            embedding_dim=hp.Int('embedding_dim', min_value=16, max_value=128, step=16),
        )
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        epochs = hp.Int('epochs', min_value=5, max_value=50, step=5)  # Defina o intervalo de épocas a serem testadas
        return model

tuner = RandomSearch(
    MatrixFactorizationHyperModel(300, 1000),
    objective='val_loss', 
    max_trials=20, 
    executions_per_trial=3,
    directory='model_tuning',
    project_name='MatrixFactorization'
)

df = pd.read_excel("X.xlsx", index_col=0)
# df = pd.read_excel("X-u5-i10_semindices.xlsx", index_col=0)
dados = df.fillna(0).values
X, y = np.nonzero(dados)
ratings = dados[X, y]
avaliacoes_tuplas = list(zip(X, y, ratings))

X_data = np.array([[usuario, item] for usuario, item, _ in avaliacoes_tuplas])
y_data = np.array([rating for _, _, rating in avaliacoes_tuplas])

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Realiza a busca pelo melhor modelo
tuner.search(x=X_train, y=y_train, epochs=10, validation_data=(X_val, y_val))

# Obtém os melhores hiperparâmetros
num_trials = 5  # Número de melhores conjuntos para recuperar
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=num_trials)

# Exibe todos os valores dos hiperparâmetros para cada conjunto
for idx, hp in enumerate(best_hyperparameters):
    print(f"Conjunto {idx + 1}:")
    for key, value in hp.values.items():
        print(f"  {key}: {value}")
