# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python badge" />
  <img src="https://img.shields.io/badge/TensorFlow%20Extended-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow Extended badge" />
  <img src="https://img.shields.io/badge/Apache%20Beam-0052CC?style=flat-square" alt="Apache Beam badge" />
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white" alt="Prometheus badge" />
  <img src="https://img.shields.io/badge/Grafana-F46800?style=flat-square&logo=grafana&logoColor=white" alt="Grafana badge" />
</p>

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

**Status terverifikasi per 27 Oktober 2026: deployment Railway saat ini belum berjalan.** Pemeriksaan layanan menemukan deployment terakhir berstatus `FAILED`; log build yang tersimpan menunjukkan konflik dependensi antara `tfx==1.14.0` dan `tensorflow==2.12.0`. Konflik pin TensorFlow telah diperbaiki di `requirements.txt` ke versi 2.13.0 untuk mengikuti rentang TFX tersebut.

Konfigurasi layanan Railway juga sudah diarahkan ke `Dockerfile` yang ada di repositori. Namun, deployment ulang belum berhasil dijalankan: Railway mengembalikan pesan bahwa masa uji coba akun telah berakhir dan meminta pemilihan paket. Jadi jangan menganggap domain atau endpoint di bawah sudah berfungsi saat ini; setelah akses deployment tersedia, jalankan build baru dan uji health/model endpoint sebelum menandainya aktif.

Model dirancang untuk di-serve dengan **TensorFlow Serving**, bukan Flask/FastAPI. `Dockerfile` memuat model dari `serving_model/` dan membuka REST API pada port yang disediakan Railway. Konfigurasi metrik ada di `config/monitoring.config`.

### Pemulihan deployment

1. Pastikan paket Railway aktif atau pindahkan layanan ke platform hosting yang dapat menjalankan image Docker ini.
2. Build ulang dari branch `main` agar perubahan dependensi dan konfigurasi Dockerfile diterapkan.
3. Pastikan build berhasil dan instance tetap hidup.
4. Uji endpoint metadata model, lalu uji prediksi dengan request `POST` yang berisi payload TensorFlow Serving yang valid.
5. Verifikasi endpoint metrik Prometheus dan catat hasil pengujian yang benar-benar berjalan.

### Endpoints yang direncanakan

| Endpoint | Method | Tujuan |
| --- | --- | --- |
| `/v1/models/wine-quality` | GET | Metadata/status model; harus diuji setelah deployment pulih. |
| `/v1/models/wine-quality:predict` | POST | Prediksi melalui TensorFlow Serving; membutuhkan body JSON sesuai signature model. |
| `/monitoring/prometheus/metrics` | GET | Metrik Prometheus bawaan TensorFlow Serving; harus diuji setelah deployment pulih. |

**Domain yang pernah dikonfigurasi:** `https://wine-quality-mlops-production.up.railway.app`. Domain ini belum merupakan bukti bahwa aplikasi sedang live.

## Monitoring Prometheus

### Prometheus Metrics yang Dipantau

| Metric                                 | Tipe          | Deskripsi                                                 |
| -------------------------------------- | ------------- | --------------------------------------------------------- |
| `:tensorflow:core:graph_runs`          | Counter       | Jumlah eksekusi graph/inference TensorFlow yang tercatat. |
| `:tensorflow:core:direct_session_runs` | Counter       | Jumlah eksekusi session TensorFlow.                       |
| `process_*`                            | Gauge/Counter | Metrik proses server.                                     |

Pada pengujian yang didokumentasikan, query
`:tensorflow:core:graph_runs` mencatat nilai `2` untuk deployment Railway
`wine-quality-mlops-production.up.railway.app`. Metric `:tensorflow:serving:request_count` tidak tersedia
pada versi image TF Serving yang digunakan, sehingga jangan gunakan query itu.

### Cara Menjalankan Monitoring Lokal

1. Copy template environment dan set kata sandi admin Grafana yang panjang serta unik:

```bash
cp .env.example .env
# Edit .env dan isi GRAFANA_ADMIN_PASSWORD dengan kata sandi unik.
```

Jangan commit file `.env`. Compose akan menolak startup ketika `GRAFANA_ADMIN_PASSWORD` kosong—tidak ada kredensial default `admin/admin`.

2. Jalankan layanan:

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100
```

3. Buka dashboard lokal: Prometheus di http://localhost:9090 dan Grafana di http://localhost:3000. Port hanya dipublikasikan ke loopback host secara default, bukan ke seluruh jaringan. Ubah port melalui `.env` jika diperlukan.

### Setup Grafana Dashboard

1. Buka http://localhost:3000 dan masuk dengan username/password yang diset di `.env`.
2. Add Data Source → Prometheus → URL: http://prometheus:9090.
3. Import dashboard atau buat panel baru dengan query `:tensorflow:core:graph_runs` untuk menghitung eksekusi inference TF Serving.

Compose menggunakan konfigurasi Prometheus lokal yang men-scrape layanan `tf-serving:8501`. Konfigurasi `monitoring/prometheus.yml` tetap digunakan oleh image monitoring untuk target deployment Railway.
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
│   ├── prometheus.yml            # Prometheus config for Railway target
│   └── prometheus.local.yml      # Prometheus config for local Docker Compose
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
cp .env.example .env  # edit GRAFANA_ADMIN_PASSWORD first
docker compose up -d --build
```
