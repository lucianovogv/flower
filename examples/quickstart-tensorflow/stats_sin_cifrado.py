#!/usr/bin/env python
"""
stats_media_std.py – Calcula media y desviación estándar de métricas
para el escenario SIN CIFRADO a partir de múltiples ejecuciones (seeds).
"""

import pandas as pd
from pathlib import Path
import re

# Ruta base de resultados
RESULTS_DIR = Path("results")

# Patrón para ficheros sin cifrado: history_federated_{R}-rounds_run{seed}.csv
pattern = re.compile(r"history_federated_(\d+)-rounds_run\d+\.csv")

# Diccionario para guardar datos agrupados por nº de rondas
grouped_data = {}

# Buscar todos los CSV que coincidan
for csv_file in RESULTS_DIR.glob("history_federated_*_run*.csv"):
    match = pattern.match(csv_file.name)
    if match:
        num_rounds = int(match.group(1))
        df = pd.read_csv(csv_file)

        # Nos quedamos con métricas finales (última fila)
        last_row = df.iloc[-1]

        acc = last_row["Accuracy"]
        loss = last_row["Loss"]
        total_time = last_row["total_time_s"]

        if num_rounds not in grouped_data:
            grouped_data[num_rounds] = {
                "Accuracy": [],
                "Loss": [],
                "total_time_s": [],
            }

        grouped_data[num_rounds]["Accuracy"].append(acc)
        grouped_data[num_rounds]["Loss"].append(loss)
        grouped_data[num_rounds]["total_time_s"].append(total_time)

# Calcular medias y std
rows = []
for num_rounds, metrics in sorted(grouped_data.items()):
    n_runs = len(metrics["Accuracy"])

    acc_mean = pd.Series(metrics["Accuracy"]).mean()
    acc_std = pd.Series(metrics["Accuracy"]).std()

    loss_mean = pd.Series(metrics["Loss"]).mean()
    loss_std = pd.Series(metrics["Loss"]).std()

    time_mean = pd.Series(metrics["total_time_s"]).mean()
    time_std = pd.Series(metrics["total_time_s"]).std()

    rows.append({
        "num_rounds": num_rounds,
        "n_runs": n_runs,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "time_mean": time_mean,
        "time_std": time_std
    })

# Crear DataFrame resumen
df_summary = pd.DataFrame(rows)

# Guardar a CSV
output_file = RESULTS_DIR / "stats_resumen_sin_cifrado.csv"
df_summary.to_csv(output_file, index=False)

print(f"✅ Resumen guardado en {output_file}")
print(df_summary.round(6).to_string(index=False))
