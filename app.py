"""Flask app untuk serving model Wine Quality dengan Prometheus metrics."""

import os
import json
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total jumlah prediction request",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latency prediction request dalam detik",
    ["endpoint"],
)
PREDICTION_RESULT = Counter(
    "prediction_result_total",
    "Total hasil prediksi berdasarkan label",
    ["label"],
)

# ── Load model ────────────────────────────────────────────────────────────────
SERVING_MODEL_DIR = os.environ.get("SERVING_MODEL_DIR", "serving_model")


def load_model():
    """Memuat model TensorFlow dari serving directory."""
    versions = [
        v for v in os.listdir(SERVING_MODEL_DIR)
        if os.path.isdir(os.path.join(SERVING_MODEL_DIR, v))
    ]
    if not versions:
        raise FileNotFoundError("Tidak ada model di serving_model directory.")
    latest = sorted(versions)[-1]
    model_path = os.path.join(SERVING_MODEL_DIR, latest)
    return tf.saved_model.load(model_path)


model = load_model()
infer = model.signatures["serving_default"]

FEATURE_KEYS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "pH", "sulphates", "alcohol",
]


def build_tf_example(data: dict) -> bytes:
    """Mengubah dict fitur menjadi serialized tf.Example."""
    feature = {}
    for key in FEATURE_KEYS:
        value = float(data[key])
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Wine Quality Prediction API"})


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint prediksi kualitas wine."""
    start = time.time()
    try:
        data = request.get_json(force=True)
        serialized = build_tf_example(data)
        input_tensor = tf.constant([serialized])
        output = infer(examples=input_tensor)

        prob = float(list(output.values())[0].numpy()[0][0])
        label = "good" if prob >= 0.5 else "bad"

        PREDICTION_RESULT.labels(label=label).inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

        return jsonify({"probability": round(prob, 4), "label": label})

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint Prometheus metrics."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
