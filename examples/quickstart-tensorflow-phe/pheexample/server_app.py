"""pheexample: Flower/TensorFlow + python-phe (Paillier)"""

from typing import List, Tuple
import os, random
import numpy as np
import tensorflow as tf

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from pheexample.task import load_model

def _set_seed(seed: int):
    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accs = [n * m["accuracy"] for n, m in metrics]
    nums = [n for n, _ in metrics]
    return {"accuracy": sum(accs) / sum(nums)}

def server_fn(context: Context):
    seed = int(context.run_config.get("seed", 42))
    _set_seed(seed)
    os.environ["SEED_VALUE"] = str(seed)
    print(f"🔹 [SERVER PHE] seed = {seed}")

    parameters = ndarrays_to_parameters(load_model().get_weights())

    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda m: {k: sum(x.get(k,0.0) for _,x in m)/len(m) for k in (m[0][1].keys() if m else [])},
    )

    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
