"""Transform module untuk Wine Quality pipeline."""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "quality"

FEATURE_KEYS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def transformed_name(key):
    """Menambahkan suffix _xf pada nama fitur."""
    return key + "_xf"


def preprocessing_fn(inputs):
    """Fungsi preprocessing untuk normalisasi fitur dan binarisasi label."""
    outputs = {}

    for key in FEATURE_KEYS:
        feature = tf.cast(inputs[key], tf.float32)
        outputs[transformed_name(key)] = tft.scale_to_z_score(feature)

    label_dense = tf.cast(inputs[LABEL_KEY], tf.int64)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.greater_equal(label_dense, 6), tf.int64
    )

    return outputs
