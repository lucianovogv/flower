#!/usr/bin/env python
"""
run_and_plot.py – Ejecuta FL y genera PNG/CSV/JSON
Compatible con Flower 1.19 (Quickstart TensorFlow, sin cifrado homomórfico)
"""

import argparse
import json
import csv
import importlib
import time
import os
from pathlib import Path

# tomllib o tomli según versión
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters
import matplotlib.pyplot as plt

# ── CLI ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Federated Learning and plot results"
)
parser.add_argument(
    "--rounds",
    type=int,
    default=None,
    help="Override num-server-rounds (si no, lee de pyproject.toml)",
)
args = parser.parse_args()

# ── Leer num-server-rounds ─────────────────────────────────────────
if args.rounds is None:
    with open("pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    num_rounds = cfg["tool"]["flwr"]["app"]["quickstart-tensorflow/tfexample/server_app.py:app"]["run-config"]["num-server-rounds"]
else:
    num_rounds = args.rounds

print(f"👉  Ejecutando simulación con {num_rounds} rondas")

# ── Importar app modules -------------------------------------------
client_module = importlib.import_module("tfexample.client_app")
server_module = importlib.import_module("tfexample.server_app")
task_module = importlib.import_module("tfexample.task")

client_fn = client_module.client_fn
weighted_average = server_module.weighted_average
load_model = task_module.load_model
get_weights = task_module.get_weights

# ── Estrategia ------------------------------------------------------
model = load_model()
weights = get_weights(model)
initial_parameters = ndarrays_to_parameters(weights)

from flwr.server.strategy import FedAvg

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
    initial_parameters=initial_parameters,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# ── Guardar resultados ---------------------------------------------
Path("results").mkdir(exist_ok=True)

# Obtener seed desde variable de entorno o pyproject
if "SEED_VALUE" in os.environ:
    seed = int(os.environ["SEED_VALUE"])
else:
    with open("pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    seed = int(cfg["tool"]["flwr"]["app"]["quickstart-tensorflow/tfexample/server_app.py:app"]["run-config"].get("seed", 42))

os.environ["SEED_VALUE"] = str(seed)

# ── Medir tiempo total ----------------------------------------------
start_time = time.time()

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=10,
    config=ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)

total_training_time = round(time.time() - start_time, 2)
print(f"⏱️  Tiempo total de entrenamiento: {total_training_time} s")

# ── Extraer métricas -----------------------------------------------
loss = [l for _, l in history.losses_distributed]
accuracy = [
    v for _, v in history.metrics_distributed.get(
        "accuracy",
        [(r, 0.0) for r in range(1, len(history.losses_distributed) + 1)]
    )
]
rounds = list(range(1, len(loss) + 1))

# Nombres de archivo con seed para no sobrescribir
json_path = f"results/history_federated_{num_rounds}-rounds_run{seed}.json"
csv_path = f"results/history_federated_{num_rounds}-rounds_run{seed}.csv"
png_path = f"results/resultados_federated_{num_rounds}rondas_run{seed}.png"

with open(json_path, "w") as f:
    json.dump({
        "loss": loss,
        "accuracy": accuracy,
        "total_time_s": total_training_time
    }, f, indent=2)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Loss", "Accuracy", "total_time_s"])
    writer.writerows(zip(rounds, loss, accuracy, [total_training_time] * num_rounds))

print(f"✅  Histórico guardado en {json_path} y {csv_path}")

# ── Graficar --------------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(rounds, loss, marker="o", color="red")
plt.title(f"Loss ({num_rounds} rounds)")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(rounds, accuracy, marker="o", color="green")
plt.title(f"Accuracy ({num_rounds} rounds)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.show()

print(f"✅  Gráfica guardada en {png_path}")
