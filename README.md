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
→ Dense(32, relu) → Dropout(0.2)
→ Dense(16, relu) → Dropout(0.2)
→ Dense(1, sigmoid)
```

### Hyperparameter Tuning (Tuner)

- Method: RandomSearch (5 trials)
- Search space: units [32,64,96,128], dropout [0.1-0.5], learning_rate [1e-2, 1e-3, 1e-4]
- Best result dari Tuner: `units=32`, `dropout_rate=0.2`, `learning_rate=0.01`

### Metrik Evaluasi

- Binary Accuracy (threshold: ≥ 0.70)
- AUC (Area Under Curve)

---

## Performa Model

Performa model dievaluasi oleh komponen Evaluator TFX. Berdasarkan output training yang tersimpan pada notebook:

- **Evaluator Binary Accuracy:** `0.7207344770431519`
- **Evaluator AUC:** `0.7961001728677963`
- **Best Tuner validation Binary Accuracy:** `0.7291507124900818`

Model hanya di-push ke serving directory jika Binary Accuracy Evaluator memenuhi threshold `≥ 0.70`.

---

## Deployment dan Monitoring

Model di-deploy dengan **TensorFlow Serving**, bukan Flask/FastAPI. `Dockerfile` menggunakan `tensorflow/serving:latest`, memuat SavedModel dari `serving_model/`, dan menjalankan REST API pada port yang disediakan Railway. Konfigurasi Prometheus bawaan TF Serving ada di `config/monitoring.config`.

### Deploy ke Railway

1. Push perubahan ke branch `main` GitHub.
2. Railway rebuild `Dockerfile` secara otomatis.
3. Pastikan deployment berstatus **Success**.
4. Buka endpoint metadata model untuk memverifikasi TF Serving.

### Endpoints

| Endpoint                          | Method | Deskripsi                            |
| --------------------------------- | ------ | ------------------------------------ |
| `/v1/models/wine-quality`         | GET    | Metadata dan status model.           |
| `/v1/models/wine-quality:predict` | POST   | Prediksi dengan REST API TF Serving. |
| `/monitoring/prometheus/metrics`  | GET    | Metrik Prometheus bawaan TF Serving. |

### Contoh Request

```bash
# Metadata model (GET)
curl https://wine-quality-mlops-production.up.railway.app/v1/models/wine-quality
```

Prediksi **wajib** menggunakan method `POST` dan body TF Serving. Jangan membuka
URL `:predict` langsung di browser karena browser mengirim `GET` dan TF Serving
akan mengembalikan `Malformed request`.

Gunakan [ridhorezkyanwar-testing.ipynb](./ridhorezkyanwar-testing.ipynb)
untuk mengirim request prediksi dengan format serialized `tf.Example`:

```python
result = predict(wine_sample)
print(result)
```

Fungsi `predict` pada notebook mengirim:

```text
POST /v1/models/wine-quality:predict
Content-Type: application/json
```

### Contoh Response

```json
{ "model_version_status": [{ "version": "<versi>", "state": "AVAILABLE" }] }
```

### Web App URL

`https://wine-quality-mlops-production.up.railway.app`

---

## Monitoring Prometheus

### Prometheus Metrics yang Dipantau

| Metric                                 | Tipe          | Deskripsi                                                 |
| -------------------------------------- | ------------- | --------------------------------------------------------- |
| `:tensorflow:core:graph_runs`          | Counter       | Jumlah eksekusi graph/inference TensorFlow yang tercatat. |
| `:tensorflow:core:direct_session_runs` | Counter       | Jumlah eksekusi session TensorFlow.                       |
| `process_*`                            | Gauge/Counter | Metrik proses server.                                     |

Pada pengujian yang didokumentasikan, query
`:tensorflow:core:graph_runs` mencatat nilai `2` untuk instance
`tf-serving:8501`. Metric `:tensorflow:serving:request_count` tidak tersedia
pada versi image TF Serving yang digunakan, sehingga jangan gunakan query itu.

### Cara Menjalankan Monitoring Lokal

```bash
# Jalankan TF Serving, Prometheus, dan Grafana
docker-compose up -d --build

# Akses Prometheus: http://localhost:9090
# Akses Grafana: http://localhost:3000 (admin/admin)
```

### Setup Grafana Dashboard

1. Buka http://localhost:3000
2. Login: admin / admin
3. Add Data Source → Prometheus → URL: http://prometheus:9090
4. Import dashboard atau buat panel baru dengan query:
   - `:tensorflow:core:graph_runs` - jumlah eksekusi inference TF Serving

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

### 3. Jalankan TensorFlow Serving dan monitoring

```bash
docker-compose up -d --build
```
