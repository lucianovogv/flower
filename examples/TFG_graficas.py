#!/usr/bin/env python
# TFG_graficas.py — Gráficas finales (media ± σ) para Sin cifrado, CKKS (TenSEAL) y Paillier (python-phe)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Rutas a los CSV de estadísticas (los que generaste con stats_*)
PLAIN_STATS = "quickstart-tensorflow/results/stats_resumen_sin_cifrado.csv"
CKKS_STATS  = "quickstart-tensorflow-he/results/stats_resumen_ckks.csv"
PHE_STATS   = "quickstart-tensorflow-phe/results/stats_resumen_phe.csv"

# Salidas (se guardan en el cwd)
OUT_ACC      = "accuracy_vs_rondas.png"
OUT_LOSS     = "loss_vs_rondas.png"
OUT_TTOTAL   = "tiempo_total_vs_rondas.png"
OUT_TROUND   = "tiempo_por_ronda_vs_rondas.png"
OUT_SZ       = "tamano_mensajes_cifrados.png"
OUT_ENCDEC   = "tiempos_cifrado_descifrado.png"

def load_df(path):
    df = pd.read_csv(path)
    # Asegura orden por número de rondas
    if "num_rounds" in df.columns:
        df = df.sort_values("num_rounds")
    return df.reset_index(drop=True)

def errplot(xs, mean, std, label, marker="o"):
    plt.errorbar(xs, mean, yerr=std, marker=marker, capsize=5, label=label)

# ────────────────────────────────────────────────────────────────────────────────
# Cargar stats
plain = load_df(PLAIN_STATS)
ckks  = load_df(CKKS_STATS)
phe   = load_df(PHE_STATS)

# Chequeos mínimos
for name, df in [("Sin cifrado", plain), ("CKKS", ckks), ("Paillier", phe)]:
    for col in ["num_rounds", "acc_mean", "acc_std", "loss_mean", "loss_std", "time_mean", "time_std"]:
        if col not in df.columns:
            raise RuntimeError(f"[{name}] Falta la columna '{col}' en {df}")

rondas = list(plain["num_rounds"].astype(int).values)

# ────────────────────────────────────────────────────────────────────────────────
# Accuracy
plt.figure(figsize=(11,7))
errplot(rondas, plain["acc_mean"], plain["acc_std"], "Sin cifrado", marker="o")
errplot(rondas, ckks["acc_mean"],  ckks["acc_std"],  "TenSEAL (CKKS)", marker="s")
errplot(rondas, phe["acc_mean"],   phe["acc_std"],   "python-phe (Paillier)", marker="^")
plt.title("Accuracy final vs número de rondas (media ± σ)")
plt.xlabel("Rondas"); plt.ylabel("Accuracy"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT_ACC, dpi=150); plt.close()

# Loss
plt.figure(figsize=(11,7))
errplot(rondas, plain["loss_mean"], plain["loss_std"], "Sin cifrado", marker="o")
errplot(rondas, ckks["loss_mean"],  ckks["loss_std"],  "TenSEAL (CKKS)", marker="s")
errplot(rondas, phe["loss_mean"],   phe["loss_std"],   "python-phe (Paillier)", marker="^")
plt.title("Loss final vs número de rondas (media ± σ)")
plt.xlabel("Rondas"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT_LOSS, dpi=150); plt.close()

# Tiempo total
plt.figure(figsize=(11,7))
errplot(rondas, plain["time_mean"], plain["time_std"], "Sin cifrado", marker="o")
errplot(rondas, ckks["time_mean"],  ckks["time_std"],  "TenSEAL (CKKS)", marker="s")
errplot(rondas, phe["time_mean"],   phe["time_std"],   "python-phe (Paillier)", marker="^")
plt.title("Tiempo total de entrenamiento (media ± σ)")
plt.xlabel("Rondas"); plt.ylabel("Tiempo total (s)"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT_TTOTAL, dpi=150); plt.close()

# Tiempo medio por ronda  (media(t)/r ; σ≈σ_t / r)
def per_round(df):
    r = df["num_rounds"].to_numpy(dtype=float)
    m = (df["time_mean"].to_numpy(dtype=float) / r).tolist()
    s = (df["time_std"].to_numpy(dtype=float) / r).tolist()
    return m, s

plain_pr_m, plain_pr_s = per_round(plain)
ckks_pr_m,  ckks_pr_s  = per_round(ckks)
phe_pr_m,   phe_pr_s   = per_round(phe)

plt.figure(figsize=(11,7))
errplot(rondas, plain_pr_m, plain_pr_s, "Sin cifrado", marker="o")
errplot(rondas, ckks_pr_m,  ckks_pr_s,  "TenSEAL (CKKS)", marker="s")
errplot(rondas, phe_pr_m,   phe_pr_s,   "python-phe (Paillier)", marker="^")
plt.title("Tiempo medio por ronda (media ± σ)")
plt.xlabel("Rondas"); plt.ylabel("Tiempo por ronda (s)"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT_TROUND, dpi=150); plt.close()

# Tamaño de mensajes cifrados (promedio sobre 3/10/20 rondas)
for col in ["sizeMB_mean", "sizeMB_std"]:
    if col not in ckks.columns or col not in phe.columns:
        raise RuntimeError("Faltan columnas de tamaño en stats_resumen_ckks/phe (sizeMB_mean/sizeMB_std).")

sz_ckks_mean = float(ckks["sizeMB_mean"].mean())
sz_ckks_std  = float(ckks["sizeMB_std"].mean())
sz_phe_mean  = float(phe["sizeMB_mean"].mean())
sz_phe_std   = float(phe["sizeMB_std"].mean())

plt.figure(figsize=(11,7))
labels = ["TenSEAL (CKKS)", "python-phe (Paillier)"]
plt.bar([0], [sz_ckks_mean], yerr=[sz_ckks_std], width=0.6, capsize=6, label="CKKS")
plt.bar([1], [sz_phe_mean],  yerr=[sz_phe_std], width=0.6, capsize=6, label="Paillier")
plt.xticks([0,1], labels)
plt.ylabel("Tamaño (MB)")
plt.title("Tamaño de mensajes cifrados (media ± σ)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_SZ, dpi=150); plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# Tiempo de cifrado/descifrado (media ± σ) — ESCALA LOG para que CKKS se vea
for col in ["enc_mean","enc_std","dec_mean","dec_std"]:
    if col not in ckks.columns or col not in phe.columns:
        raise RuntimeError(f"Falta columna '{col}' en stats_resumen_ckks/phe.")

enc_means = [float(ckks["enc_mean"].mean()), float(phe["enc_mean"].mean())]
enc_stds  = [float(ckks["enc_std"].mean()),  float(phe["enc_std"].mean())]
dec_means = [float(ckks["dec_mean"].mean()), float(phe["dec_mean"].mean())]
dec_stds  = [float(ckks["dec_std"].mean()),  float(phe["dec_std"].mean())]

# Para impresión de verificación
print("== Tiempos de cifrado/descifrado usados en la gráfica ==")
for lab, em, es, dm, ds in zip(labels, enc_means, enc_stds, dec_means, dec_stds):
    print(f"{lab:24s}  enc_mean={em:.6f}s  enc_std={es:.6f}s  dec_mean={dm:.6f}s  dec_std={ds:.6f}s")

# Evita log(0)
EPS = 1e-3
enc_means_plot = [max(m, EPS) for m in enc_means]
dec_means_plot = [max(m, EPS) for m in dec_means]
enc_stds_plot  = [max(s, EPS/10) for s in enc_stds]
dec_stds_plot  = [max(s, EPS/10) for s in dec_stds]

x = np.arange(2); w = 0.35
plt.figure(figsize=(12,6))
plt.yscale("log")
ymax = max(max(enc_means_plot), max(dec_means_plot)) * 1.5
plt.ylim(EPS/2, ymax)

plt.bar(x - w/2, enc_means_plot, yerr=enc_stds_plot, width=w, label="Cifrado", capsize=6)
plt.bar(x + w/2, dec_means_plot, yerr=dec_stds_plot, width=w, label="Descifrado", capsize=6)

plt.xticks(x, labels)
plt.ylabel("Tiempo (s)  [escala log]")
plt.title("Tiempo de cifrado/descifrado (media ± σ)")
plt.grid(axis="y", which="both", alpha=0.3)
plt.legend()

# Anota valores reales encima (no los “clipeados”)
def annotate(xs, vals_real, vals_plot):
    for xi, vr, vp in zip(xs, vals_real, vals_plot):
        plt.text(xi, vp*1.05, f"{vr:.4f}s", ha="center", va="bottom", fontsize=9)
annotate(x - w/2, enc_means, enc_means_plot)
annotate(x + w/2, dec_means, dec_means_plot)

plt.tight_layout(); plt.savefig(OUT_ENCDEC, dpi=150); plt.close()

print("✅ Gráficas finales generadas:")
for p in [OUT_ACC, OUT_LOSS, OUT_TTOTAL, OUT_TROUND, OUT_SZ, OUT_ENCDEC]:
    print(" •", os.path.abspath(p))
