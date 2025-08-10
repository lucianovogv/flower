"""tfexample: A Flower / TensorFlow app."""
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from tfexample.task import load_model

# ---- Aleatoriedad
import os, random
import numpy as np
import tensorflow as tf

def _set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Métricas agregadas
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [n * m["accuracy"] for n, m in metrics]
    examples = [n for n, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # 1) Seed desde pyproject (run_config); por defecto 42
    seed = int(context.run_config.get("seed", 42))
    _set_seed(seed)
    # La exponemos para los clientes
    os.environ["SEED_VALUE"] = str(seed)
    print(f"🔹 [SERVER] seed(run_config) = {seed}")

    # 2) Modelo inicial
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # 3) Estrategia
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 4) Config de servidor (rondas viene de run_config)
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
