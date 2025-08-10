"""tfexample/client_app.py – Cliente Flower para la versión sin cifrado"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tfexample.task import load_data, load_model

# =========================
# Control de aleatoriedad (cliente)
# =========================
import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Prioridad: variable de entorno SEED_VALUE (si existe), si no 42
_DEFAULT_SEED = int(os.getenv("SEED_VALUE", "42"))
set_global_seed(_DEFAULT_SEED)
print(f"🔹 [CLIENT] Seed inicial = {_DEFAULT_SEED}")

class FlowerClient(NumPyClient):
    def __init__(self, learning_rate, data, epochs, batch_size, verbose):
        self.model = load_model()
        self.lr = learning_rate
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Entrena el modelo con los datos de este cliente."""
        self.model.set_weights(parameters)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evalúa el modelo en los datos de este cliente."""
        self.model.set_weights(parameters)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

def client_fn(context: Context):
    """Construye un Client que se ejecutará en un ClientApp."""
    # Si en run_config viene una seed explícita, la aplicamos (tiene prioridad)
    seed_from_config = context.run_config.get("seed", None)
    if seed_from_config is not None:
        try:
            seed_value = int(seed_from_config)
            set_global_seed(seed_value)
            print(f"🔹 [CLIENT] Seed desde run_config = {seed_value}")
        except Exception:
            # Si no es convertible a int, mantenemos la seed anterior
            print(f"⚠️ [CLIENT] run_config['seed'] no es entero: {seed_from_config}")

    # Config de partición de datos
    partition_id = int(context.node_config.get("partition-id", 0))
    num_partitions = int(context.node_config.get("num-partitions", 1))

    # run_config
    epochs = int(context.run_config.get("local-epochs", context.run_config.get("epochs", 1)))
    batch_size = int(
        context.run_config.get(
            "batch-size",
            context.run_config.get("batch_size", context.run_config.get("batchSize", 32)),
        )
    )
    verbose = int(context.run_config.get("verbose", 0))
    learning_rate = float(context.run_config.get("learning-rate", 0.001))

    # Carga de datos para esta partición
    data = load_data(partition_id, num_partitions, batch_size)

    return FlowerClient(
        learning_rate, data, epochs, batch_size, verbose
    ).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)
