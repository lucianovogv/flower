"""tfexample/client_app.py – Cliente Flower para CKKS (TenSEAL)"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tfexample.task import load_data, load_model

import os, random, time
import numpy as np
import tensorflow as tf

import tenseal as ts


def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


# Seed base desde entorno (server/run_and_plot la exporta como SEED_VALUE)
BASE_SEED = int(os.getenv("SEED_VALUE", "42"))
set_global_seed(BASE_SEED)

# ---- Contexto CKKS (parámetros de ejemplo; ajusta a los tuyos si ya los tienes) ----
CKKS_POLYMOD = 8192
CKKS_COEFF_MOD_BITS = [40, 21, 21, 40]  # ~nivel 3
CKKS_SCALE = 2**40

CKKS_CTX = ts.context(ts.SCHEME_TYPE.CKKS, CKKS_POLYMOD, -1, CKKS_COEFF_MOD_BITS)
CKKS_CTX.generate_galois_keys()
CKKS_CTX.generate_relin_keys()
CKKS_CTX.global_scale = CKKS_SCALE
# (No existe CKKS_CTX.generate_secret_key(); la secret key ya está en el contexto)


class FlowerClient(NumPyClient):
    def __init__(self, learning_rate, data, epochs, batch_size, verbose):
        self.model = load_model()
        self.lr = learning_rate
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
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

        # === Instrumentación CKKS para métricas HE ==================
        weights = self.model.get_weights()
        flat = np.concatenate([w.ravel() for w in weights]).astype(np.float64)
        flat_list = flat.tolist()

        t0 = time.perf_counter()
        enc_vec = ts.ckks_vector(CKKS_CTX, flat_list)
        t1 = time.perf_counter()
        enc_time = t1 - t0

        enc_bytes = enc_vec.serialize()
        cipher_size_bytes = len(enc_bytes)

        t2 = time.perf_counter()
        _ = enc_vec.decrypt()  # sin argumentos
        t3 = time.perf_counter()
        dec_time = t3 - t2
        # ============================================================

        metrics = {
            "enc_time": float(enc_time),
            "dec_time": float(dec_time),
            "cipher_size_bytes": int(cipher_size_bytes),
        }
        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    # Si llega seed por run_config, úsala
    seed_from_config = context.run_config.get("seed", None)
    if seed_from_config is not None:
        try:
            s = int(seed_from_config)
            set_global_seed(s)
            os.environ["SEED_VALUE"] = str(s)
            print(f"🔹 [CLIENT CKKS] Seed desde run_config = {s}")
        except Exception:
            print(f"⚠️ [CLIENT CKKS] run_config['seed'] inválida: {seed_from_config}")

    pid = int(context.node_config.get("partition-id", 0))
    n_parts = int(context.node_config.get("num-partitions", 1))

    epochs = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 32))
    verbose = int(context.run_config.get("verbose", 0))
    learning_rate = float(context.run_config.get("learning-rate", 0.001))

    data = load_data(pid, n_parts, batch_size)
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


app = ClientApp(client_fn=client_fn)
