# quickstart-tensorflow/task.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

def load_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)

def train(model, train_data, val_data, epochs, lr):
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"],
    )
    model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=32, verbose=0)

def test(model, test_data):
    loss, accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
    return loss, accuracy

def load_data(partition_id, num_partitions, batch_size):
    # Cargar dataset de diabetes (Pima Indians)
    from sklearn.datasets import load_diabetes
    import pandas as pd
    from sklearn.datasets import fetch_openml

    # Cargar desde OpenML
    df = fetch_openml(name='diabetes', version=1, as_frame=True)
    X = df.data.to_numpy()
    y = df.target.to_numpy()
    y = (y == 'tested_positive').astype(np.float32)

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Particionar
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dividir entre clientes
    total = len(X_train)
    part_size = total // num_partitions
    start = partition_id * part_size
    end = start + part_size

    x_part = X_train[start:end]
    y_part = y_train[start:end]

    return (x_part, y_part), (X_val, y_val)
