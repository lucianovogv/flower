#!/usr/bin/env python
"""
run_and_plot.py – Entrena y genera PNG/CSV/JSON (incluye tiempo total en CSV)
Escenario: CKKS (TenSEAL) con Flower 1.19
"""

import argparse
import json
import csv
import importlib
import os
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, Metrics

# tomllib / tomli compatibility
try:
    import tomllib  # Python ≥3.11
except ModuleNotFoundError:
    import tomli as tomllib


# -------------------- Helpers para leer config -------------------- #
def _read_ckks_block():
    """Lee el bloque CKKS del pyproject.toml local."""
    with open("pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    return cfg["tool"]["flwr"]["app"]["quickstart-tensorflow-he/tfexample/server_app.py:app"]["run-config"]


def get_num_rounds():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override num-server-rounds (si no, lee de pyproject.toml)",
    )
    args = parser.parse_args()
    if args.rounds is not None:
        return args.rounds
    run_cfg = _read_ckks_block()
    return int(run_cfg["num-server-rounds"])


def get_seed():
    # Prioridad a SEED_VALUE si ya viene del entorno
    if "SEED_VALUE" in os.environ:
        return int(os.environ["SEED_VALUE"])
    # Si no, la leemos del pyproject de CKKS y la exportamos
    run_cfg = _read_ckks_block()
    return int(run_cfg.get("seed", 42))


# -------------------- Config de ejecución -------------------- #
num_rounds = get_num_rounds()
seed = get_seed()
os.environ["SEED_VALUE"] = str(seed)  # importante: que clientes la hereden
print(f"👉 Ejecutando CKKS con {num_rounds} rondas | seed = {seed}")

# -------------------- Importar módulos CKKS -------------------- #
client_mod = importlib.import_module("tfexample.client_app")
server_mod = importlib.import_module("tfexample.server_app")
task_mod = importlib.import_module("tfexample.task")

from flwr.server.strategy import FedAvg
weighted_average = server_mod.weighted_average
load_model = task_mod.load_model
get_weights = task_mod.get_weights

# Métrica agregada para fit (media simple)
def fit_metrics_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    agg: Metrics = {}
    for _, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    n = len(metrics)
    return {k: v / n for k, v in agg.items()}

# -------------------- Estrategia -------------------- #
template_model = load_model()
initial_weights = get_weights(template_model)
initial_parameters = ndarrays_to_parameters(initial_weights)

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=fit_metrics_average,
    initial_parameters=initial_parameters,
)

# -------------------- Entrenamiento -------------------- #
start_time = time.time()

history = fl.simulation.start_simulation(
    client_fn=client_mod.client_fn,
    num_clients=10,
    config=ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)

total_time = round(time.time() - start_time, 2)
print(f"⏱️  Tiempo total de entrenamiento: {total_time} s")

# -------------------- Extraer métricas -------------------- #
rounds = list(range(1, num_rounds + 1))
losses = [l for _, l in history.losses_distributed]
accuracy = [
    v for _, v in history.metrics_distributed.get(
        "accuracy",
        [(r, 0.0) for r in rounds]
    )
]

# Métricas HE reportadas en fit
fit_src = history.metrics_distributed_fit
enc_times = [v for _, v in fit_src.get("enc_time", [])]
dec_times = [v for _, v in fit_src.get("dec_time", [])]
cipher_bytes = [v for _, v in fit_src.get("cipher_size_bytes", [])]
sizes_mb = [b / (1024 * 1024) for b in cipher_bytes] if cipher_bytes else [0.0] * num_rounds

# -------------------- Guardar resultados -------------------- #
results_dir = "results"
Path(results_dir).mkdir(exist_ok=True)

# nombres con sufijo _run{seed}
json_path = f"{results_dir}/history_HE_{num_rounds}r_run{seed}.json"
csv_path  = f"{results_dir}/history_HE_{num_rounds}r_run{seed}.csv"
png_path  = f"{results_dir}/metrics_HE_{num_rounds}r_run{seed}.png"

# JSON
with open(json_path, "w") as f:
    json.dump(
        {
            "loss": losses,
            "accuracy": accuracy,
            "enc_time_s": enc_times,
            "dec_time_s": dec_times,
            "cipher_size_MB": sizes_mb,
            "total_time_s": total_time,
        },
        f,
        indent=2,
    )

# CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "loss", "acc", "enc_s", "dec_s", "size_MB", "total_time_s"])
    for i in range(num_rounds):
        enc_i = enc_times[i] if i < len(enc_times) else 0.0
        dec_i = dec_times[i] if i < len(dec_times) else 0.0
        sz_i  = sizes_mb[i] if i < len(sizes_mb) else 0.0
        writer.writerow([rounds[i], losses[i], accuracy[i], enc_i, dec_i, sz_i, total_time])

print(f"✅ Histórico guardado en {csv_path}")

# -------------------- Gráfica rápida -------------------- #
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(rounds, losses, marker="o")
plt.title("Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(rounds, accuracy, marker="o")
plt.title("Accuracy")
plt.grid(True)

plt.subplot(1, 3, 3)
if enc_times:
    plt.plot(rounds[:len(enc_times)], enc_times, marker="o", label="Enc (s)")
if dec_times:
    plt.plot(rounds[:len(dec_times)], dec_times, marker="o", label="Dec (s)")
if sizes_mb:
    plt.plot(rounds[:len(sizes_mb)], sizes_mb, marker="o", label="Size (MB)")
plt.title("HE metrics")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.show()
print(f"✅ Gráfica guardada en {png_path}")
