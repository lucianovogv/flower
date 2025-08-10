#!/usr/bin/env python
# stats_ckks.py – Media y desviación estándar para CKKS (TenSEAL)

import glob
from pathlib import Path
import pandas as pd

# Si ejecutas desde quickstart-tensorflow-he/, la carpeta es "results"
BASE_DIR = Path("results")
RONDAS = [3, 10, 20]

# Asegura carpeta destino
BASE_DIR.mkdir(exist_ok=True)

def pick(files):
    return sorted(files)

def get_col(df, names):
    cmap = {c.lower(): c for c in df.columns}
    alias = {
        "accuracy": ["accuracy", "acc"],
        "loss": ["loss", "val_loss"],
        "total_time_s": ["total_time_s", "time_s", "runtime_s"],
        "enc_s": ["enc_s", "enc_time"],
        "dec_s": ["dec_s", "dec_time"],
        "size_mb": ["size_mb", "cipher_size_mb", "size (mb)"],
    }
    for n in names:
        for cand in alias.get(n.lower(), [n]):
            k = cand.lower()
            if k in cmap:
                return df[cmap[k]]
    raise KeyError(f"No encuentro {names} en columnas {list(df.columns)}")

rows = []
for r in RONDAS:
    pattern = str(BASE_DIR / f"history_HE_{r}r_run*.csv")
    files = pick(glob.glob(pattern))
    if not files:
        print(f"ℹ️  No hay ficheros para {r} rondas con patrón {pattern}")
    accs, losses, tts, encs, decs, sizes = [], [], [], [], [], []
    for f in files:
        try:
            df = pd.read_csv(f)
            accs.append(float(get_col(df, ["accuracy","acc"]).dropna().iloc[-1]))
            losses.append(float(get_col(df, ["loss"]).dropna().iloc[-1]))
            # tiempo total medio por fichero
            tts.append(float(get_col(df, ["total_time_s"]).mean()))
            # opcionales
            try:
                encs.append(float(get_col(df, ["enc_s","enc_time"]).mean()))
            except Exception:
                pass
            try:
                decs.append(float(get_col(df, ["dec_s","dec_time"]).mean()))
            except Exception:
                pass
            try:
                sizes.append(float(get_col(df, ["size_mb","cipher_size_mb","size (mb)"]).mean()))
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️  Salto {f}: {e}")

    def mstd(a):
        s = pd.Series(a)
        return (s.mean(), s.std()) if len(s) else (float("nan"), float("nan"))

    acc_m, acc_s = mstd(accs)
    loss_m, loss_s = mstd(losses)
    tt_m, tt_s = mstd(tts)
    enc_m, enc_s = mstd(encs)
    dec_m, dec_s = mstd(decs)
    sz_m,  sz_s  = mstd(sizes)

    rows.append({
        "num_rounds": r,
        "n_runs": len(files),
        "acc_mean": acc_m,  "acc_std": acc_s,
        "loss_mean": loss_m,"loss_std": loss_s,
        "time_mean": tt_m,  "time_std": tt_s,
        "enc_mean": enc_m,  "enc_std": enc_s,
        "dec_mean": dec_m,  "dec_std": dec_s,
        "sizeMB_mean": sz_m,"sizeMB_std": sz_s,
    })

out = pd.DataFrame(rows)
out_path = BASE_DIR / "stats_resumen_ckks.csv"
out.to_csv(out_path, index=False)
print(f"✅ Resumen CKKS guardado en {out_path}")
print(out.round(6).to_string(index=False))
