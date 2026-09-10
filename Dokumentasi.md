# Submission 1: Wine Quality Classification

Nama: Ridho Rezky Anwar

Username dicoding: ridhorezkyanwar

|                         | Deskripsi                                                                                                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset                 | [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) dari UCI Machine Learning Repository. Dataset berisi 6.497 data wine merah dan putih, 11 fitur kimiawi, dan label `quality`.                                  |
| Masalah                 | Menentukan apakah wine tergolong `good` atau `bad` berdasarkan komposisi kimianya agar penilaian kualitas dapat diotomatisasi.                                                                                                             |
| Solusi machine learning | Model klasifikasi biner TensorFlow. Pipeline TFX melakukan validasi, transformasi, tuning, training, evaluasi, dan menghasilkan SavedModel bila model memenuhi threshold.                                                                  |
| Metode pengolahan       | Sebelas fitur numerik dinormalisasi dengan z-score melalui `tft.scale_to_z_score`. Label dibinarisasi: `quality >= 6` menjadi `1` (`good`) dan sisanya `0` (`bad`).                                                                        |
| Arsitektur model        | Input 11 fitur, Dense(32, ReLU), Dropout(0,2), Dense(32, ReLU), Dropout(0,2), dan Dense(1, sigmoid). Hyperparameter terbaik dari RandomSearch lima trial adalah `units=32`, `dropout_rate=0.2`, dan `learning_rate=0.01`.                  |
| Metrik evaluasi         | Binary Accuracy dan AUC. Model hanya dapat di-push bila Binary Accuracy memenuhi threshold minimal 0,70.                                                                                                                                   |
| Performa model          | Evaluator menghasilkan Binary Accuracy `0.7207344770431519` dan AUC `0.7961001728677963`; best Tuner validation Binary Accuracy `0.7291507124900818`. Model memenuhi threshold Binary Accuracy minimal `0,70` dan mendapat status blessed. |
| Opsi deployment         | SavedModel dijalankan menggunakan **TensorFlow Serving** dalam container `tensorflow/serving:latest` dan dideploy ke Railway. REST API TF Serving menggunakan port Railway (`$PORT`).                                                      |
| Web app                 | [TF Serving model metadata](https://wine-quality-mlops-production.up.railway.app/v1/models/wine-quality)                                                                                                                                   |
| Monitoring              | TF Serving mengekspos metrik Prometheus bawaan di `/monitoring/prometheus/metrics`. Prometheus melakukan scrape endpoint ini setiap lima detik melalui konfigurasi `monitoring/prometheus.yml`.                                            |

---

## Bukti Deployment

Setelah Railway menyelesaikan redeploy image TF Serving, ambil screenshot respons endpoint berikut dan simpan sebagai `ridhorezkyanwar-deployment.png`:

```text
https://wine-quality-mlops-production.up.railway.app/v1/models/wine-quality
```

Respons yang diharapkan memuat status model, nama `wine-quality`, dan versi model yang dimuat. Screenshot deployment sebelumnya yang menampilkan Flask tidak digunakan sebagai bukti deployment TF Serving.

---

## Bukti Monitoring

Konfigurasi Prometheus pada submission melakukan scrape deployment TF Serving di Railway melalui HTTPS. Target yang digunakan adalah
`wine-quality-mlops-production.up.railway.app`, dengan metrics path
`/monitoring/prometheus/metrics`.

Untuk menjalankan stack monitoring lokal setelah Docker tersedia:

```bash
docker-compose up -d --build
```

1. Kirim beberapa request prediksi dengan method `POST` ke
   `http://localhost:8501/v1/models/wine-quality:predict` menggunakan
   [ridhorezkyanwar-testing.ipynb](./ridhorezkyanwar-testing.ipynb).
   Endpoint `:predict` tidak dapat diuji dengan membuka URL di browser karena
   browser mengirim method `GET`.
2. Buka `http://localhost:9090/graph`. Prometheus akan menampilkan target
   Railway pada halaman `http://localhost:9090/targets` dengan status `UP`
   jika deployment dapat dijangkau.
3. Jalankan query `:tensorflow:core:graph_runs` dan pilih tab **Graph**.
4. Simpan screenshot grafik time series Prometheus sebagai `ridhorezkyanwar-monitoring.png`. Pada pengujian yang didokumentasikan, `:tensorflow:core:graph_runs` mencapai nilai `2`. Metric `:tensorflow:serving:request_count` tidak tersedia pada image TF Serving yang digunakan.

Endpoint metrik deployment yang dapat diverifikasi langsung adalah:

```text
https://wine-quality-mlops-production.up.railway.app/monitoring/prometheus/metrics
```

---

## Instruksi Verifikasi

1. Buka endpoint metadata: `GET /v1/models/wine-quality`.
2. Verifikasi metrik: `GET /monitoring/prometheus/metrics`.
3. Kirim prediksi melalui endpoint standar TF Serving: `POST /v1/models/wine-quality:predict`.
4. Lihat `Dockerfile`, `tf_serving_entrypoint.sh`, `config/monitoring.config`, dan `monitoring/prometheus.yml` untuk konfigurasi deployment dan monitoring.
