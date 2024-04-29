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

# Gerando dados
num_users, num_items, embedding_dim, num_samples = 300, 1000, 32, 10000
X = np.random.randint(0, num_users, size=(num_samples, 1))
item_ids = np.random.randint(0, num_items, size=(num_samples, 1))
X = np.hstack((X, item_ids))
y = np.random.randint(1, 6, size=(num_samples, 1))

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciando e treinando o modelo
model = MatrixFactorizationNN(num_users, num_items, embedding_dim)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Métricas de erro
predictions_test = model.predict(X_test)
mse = mean_squared_error(y_test, predictions_test)
mae = mean_absolute_error(y_test, predictions_test)

# Exibindo métricas
print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")

# Exibir predições para o conjunto de teste
test_results = pd.DataFrame({"Real": y_test.flatten(), "Predicted": predictions_test.flatten()})
print("Sample test predictions vs. real:\n", test_results.sample(10))

# Matriz completa de predições
all_predictions_df = model.predict_all()
print("Complete predictions matrix:\n", all_predictions_df)
