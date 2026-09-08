"""Script untuk menggabungkan red dan white wine menjadi wine_quality.csv."""

import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def read_csv_semicolon(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def normalize_header(header):
    return header.strip().replace(" ", "_")


red_path = os.path.join(DATA_DIR, "winequality-red.csv")
white_path = os.path.join(DATA_DIR, "winequality-white.csv")
output_path = os.path.join(DATA_DIR, "wine_quality.csv")

red_rows = read_csv_semicolon(red_path)
white_rows = read_csv_semicolon(white_path)

all_rows = red_rows + white_rows

# Normalize header
raw_headers = list(all_rows[0].keys())
norm_headers = [normalize_header(h) for h in raw_headers]

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(norm_headers)
    for row in all_rows:
        writer.writerow([row[h] for h in raw_headers])

print(f"Berhasil! Total baris: {len(all_rows)}")
print(f"Kolom: {norm_headers}")

# Hitung distribusi quality
quality_counts = {}
good = 0
bad = 0
for row in all_rows:
    q = int(float(row["quality"]))
    quality_counts[q] = quality_counts.get(q, 0) + 1
    if q >= 6:
        good += 1
    else:
        bad += 1

print(f"\nDistribusi quality:")
for k in sorted(quality_counts):
    print(f"  {k}: {quality_counts[k]}")
print(f"\nLabel binarisasi:")
print(f"  good (>=6): {good}")
print(f"  bad  (<6) : {bad}")
print(f"\nDisimpan ke: {output_path}")
