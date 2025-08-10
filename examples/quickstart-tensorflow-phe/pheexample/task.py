import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model

def load_data(partition_id, num_partitions, batch_size):
    df = fetch_openml(name='diabetes', version=1, as_frame=True)
    X = df.data.to_numpy()
    y = (df.target.to_numpy() == 'tested_positive').astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    total = len(X_train)
    part_size = total // num_partitions
    start = partition_id * part_size
    end = start + part_size

    x_part = X_train[start:end]
    y_part = y_train[start:end]

    return (x_part, y_part), (X_val, y_val)

def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)
