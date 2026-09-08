# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

## Wine Quality Classification

**Nama:** Ridho Rezky Anwar  
**Username Dicoding:** ridhorezkyanwar

---

## Informasi Dataset

| Atribut      | Detail                                               |
| ------------ | ---------------------------------------------------- |
| Nama         | Wine Quality Dataset                                 |
| Sumber       | UCI Machine Learning Repository                      |
| URL          | https://archive.ics.uci.edu/ml/datasets/wine+quality |
| Jumlah Data  | 6497 baris (red: 1599, white: 4898)                  |
| Jumlah Fitur | 11 fitur numerik                                     |
| Label        | quality (skor 0-10)                                  |

**Fitur yang digunakan:**

- fixed_acidity, volatile_acidity, citric_acid
- residual_sugar, chlorides
- free_sulfur_dioxide, total_sulfur_dioxide
- density, pH, sulphates, alcohol

---

## Persoalan yang Ingin Diselesaikan

Industri wine membutuhkan cara otomatis untuk menilai kualitas wine berdasarkan komposisi kimianya tanpa harus mengandalkan penilaian manual oleh sommelier yang memakan waktu dan biaya tinggi.

**Pertanyaan bisnis:** Apakah sebuah wine termasuk kualitas **good** atau **bad** berdasarkan 11 parameter kimia?

---

## Solusi Machine Learning

- **Task:** Binary Classification
- **Label:** quality ≥ 6 → `good (1)`, quality < 6 → `bad (0)`
- **Target Performa:** Binary Accuracy ≥ 70%
- **Orchestrator:** Apache Beam via TFX
- **Platform Deployment:** Railway
- **Monitoring:** Prometheus + Grafana

---

## Metode Pengolahan Data & Arsitektur Model

### Preprocessing (Transform)

- Z-score normalization untuk semua 11 fitur numerik menggunakan `tft.scale_to_z_score`
- Binarisasi label: quality ≥ 6 → 1, quality < 6 → 0

### Arsitektur Model

```
Input (11 fitur)
→ Dense(64, relu) → Dropout(0.3)
→ Dense(32, relu) → Dropout(0.3)
→ Dense(1, sigmoid)
```

### Hyperparameter Tuning (Tuner)

- Method: RandomSearch (5 trials)
- Search space: units [32,64,96,128], dropout [0.1-0.5], learning_rate [1e-2, 1e-3, 1e-4]

### Metrik Evaluasi

- Binary Accuracy (threshold: ≥ 0.70)
- AUC (Area Under Curve)

---

## Performa Model

Performa model dievaluasi oleh komponen Evaluator TFX:

- **Binary Accuracy:** ≥ 0.70 (threshold untuk model di-push)
- **AUC:** dilihat dari output Evaluator setelah pipeline selesai

Model hanya di-push ke serving directory jika memenuhi threshold Binary Accuracy ≥ 0.70.

---

## Deployment

### Platform

**Railway** - https://railway.app

### Cara Deploy

1. Push kode ke GitHub repository
2. Connect repository ke Railway
3. Set environment variable: `SERVING_MODEL_DIR=serving_model`
4. Railway otomatis build dari Dockerfile dan deploy

### Endpoints

| Endpoint   | Method | Deskripsi              |
| ---------- | ------ | ---------------------- |
| `/`        | GET    | Info API               |
| `/predict` | POST   | Prediksi kualitas wine |
| `/health`  | GET    | Health check           |
| `/metrics` | GET    | Prometheus metrics     |

### Contoh Request

```bash
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.28,
    "citric_acid": 0.34,
    "residual_sugar": 1.2,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 35.0,
    "total_sulfur_dioxide": 141.0,
    "density": 0.9940,
    "pH": 3.42,
    "sulphates": 0.68,
    "alcohol": 12.5
  }'
```

### Contoh Response

```json
{
  "probability": 0.8234,
  "label": "good"
}
```

### Web App URL

> Ganti dengan URL Railway Anda setelah deploy: `https://your-app.railway.app`

---

## Monitoring

### Prometheus Metrics yang Dipantau

| Metric                               | Tipe      | Deskripsi                            |
| ------------------------------------ | --------- | ------------------------------------ |
| `prediction_requests_total`          | Counter   | Total jumlah request prediksi        |
| `prediction_request_latency_seconds` | Histogram | Latency setiap request               |
| `prediction_result_total`            | Counter   | Distribusi hasil prediksi (good/bad) |

### Cara Menjalankan Monitoring Lokal

```bash
# Jalankan Flask app terlebih dahulu
python app.py

# Jalankan Prometheus + Grafana
docker-compose up -d

# Akses Prometheus: http://localhost:9090
# Akses Grafana: http://localhost:3000 (admin/admin)
```

### Setup Grafana Dashboard

1. Buka http://localhost:3000
2. Login: admin / admin
3. Add Data Source → Prometheus → URL: http://prometheus:9090
4. Import dashboard atau buat panel baru dengan query:
   - `rate(prediction_requests_total[1m])` - request rate
   - `histogram_quantile(0.95, prediction_request_latency_seconds_bucket)` - P95 latency
   - `prediction_result_total` - distribusi prediksi

---

## Struktur Proyek

```
ProyekPengembangandanPengoperasianSistemMachineLearning/
├── ridhorezkyanwar-pipeline/     # TFX pipeline artifacts
├── modules/
│   ├── transform.py              # Preprocessing module
│   ├── trainer.py                # Training module
│   └── tuner.py                  # Hyperparameter tuning module
├── monitoring/
│   ├── Dockerfile                # Prometheus Dockerfile
│   ├── prometheus.config         # Prometheus config
│   └── prometheus.yml            # Prometheus scrape config
├── data/
│   └── wine_quality.csv          # Dataset
├── serving_model/                # Pushed model (generated)
├── notebook.ipynb                # Pipeline notebook
├── ridhorezkyanwar-testing.ipynb # Testing notebook
├── pipeline.py                   # Pipeline script
├── app.py                        # Flask serving app
├── requirements.txt
├── Dockerfile                    # App Dockerfile
├── docker-compose.yml            # Prometheus + Grafana
└── README.md
```

---

## Cara Menjalankan

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Jalankan Pipeline

```bash
python pipeline.py
# atau buka notebook.ipynb dan jalankan semua cell
```

### 3. Jalankan Flask App Lokal

```bash
python app.py
```

### 4. Jalankan Monitoring

```bash
docker-compose up -d
```
