"""pheexample/client_app.py – Cliente Flower para Paillier (python-phe)"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pheexample.task import load_data, load_model

import os, random, time
import numpy as np
import tensorflow as tf

from phe import paillier

def set_seed(s: int):
    np.random.seed(s); tf.random.set_seed(s); random.seed(s)

BASE_SEED = int(os.getenv("SEED_VALUE", "42"))
set_seed(BASE_SEED)

# ----- Claves Paillier globales (demo/medición) -----
PUBKEY, PRIVKEY = paillier.generate_paillier_keypair(n_length=2048)
SCALE = 1e6            # para pasar float -> int
MAX_ELEMS = 5000       # limitar nº elementos para medir (tiempo/tamaño)

class FlowerClient(NumPyClient):
    def __init__(self, lr, data, epochs, batch_size, verbose):
        self.model = load_model()
        self.lr = lr
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
            self.x_train, self.y_train,
            epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose
        )

        # === Instrumentación Paillier ===============================
        # A) vectorizamos pesos y los cuantizamos a enteros
        weights = self.model.get_weights()
        flat = np.concatenate([w.ravel() for w in weights]).astype(np.float64)
        flat = np.clip(flat, -1e6, 1e6)                  # evitar overflow raro
        ints = np.round(flat * SCALE).astype(np.int64)

        K = int(min(MAX_ELEMS, len(ints)))
        sample = ints[:K]

        # B) cifrado
        t0 = time.perf_counter()
        enc = [PUBKEY.encrypt(int(v)) for v in sample]
        t1 = time.perf_counter()
        enc_time = t1 - t0

        # C) tamaño (bytes) estimado sumando tamaño de cada ciphertext
        size_bytes = 0
        for c in enc:
            nbytes = (c.ciphertext().bit_length() + 7) // 8
            size_bytes += nbytes

        # D) descifrado
        t2 = time.perf_counter()
        _ = [PRIVKEY.decrypt(c) for c in enc]
        t3 = time.perf_counter()
        dec_time = t3 - t2
        # ============================================================

        metrics = {
            "enc_time": float(enc_time),
            "dec_time": float(dec_time),
            "cipher_size_bytes": int(size_bytes),
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
    s = context.run_config.get("seed", None)
    if s is not None:
        try:
            s = int(s)
            set_seed(s)
            os.environ["SEED_VALUE"] = str(s)
            print(f"🔹 [CLIENT PHE] seed = {s}")
        except Exception:
            pass

    pid = int(context.node_config.get("partition-id", 0))
    n_parts = int(context.node_config.get("num-partitions", 1))

    epochs = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 32))
    verbose = int(context.run_config.get("verbose", 0))
    lr = float(context.run_config.get("learning-rate", 0.001))

    data = load_data(pid, n_parts, batch_size)
    return FlowerClient(lr, data, epochs, batch_size, verbose).to_client()

app = ClientApp(client_fn=client_fn)
