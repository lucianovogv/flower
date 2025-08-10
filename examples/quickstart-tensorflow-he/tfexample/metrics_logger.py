import csv
import os

def log_metrics(filename, round_num, enc_time, dec_time, size_bytes):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["round", "enc_time", "dec_time", "size_bytes"])
        writer.writerow([round_num, enc_time, dec_time, size_bytes])
