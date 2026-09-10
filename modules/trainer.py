"""Trainer module untuk Wine Quality pipeline."""

import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.fn_args_utils import FnArgs

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


def _gzip_reader_fn(filenames):
    """Membaca file TFRecord yang dikompresi GZIP."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    """Membuat dataset dari file pattern."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def _build_keras_model(hp=None):
    """Membangun model Keras untuk klasifikasi biner."""
    units = hp.get("units") if hp else 64
    dropout_rate = hp.get("dropout_rate") if hp else 0.3
    learning_rate = hp.get("learning_rate") if hp else 0.001

    inputs = {
        transformed_name(key): tf.keras.Input(shape=(1,), name=transformed_name(key))
        for key in FEATURE_KEYS
    }

    x = tf.keras.layers.concatenate(list(inputs.values()))
    x = tf.keras.layers.Dense(units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units // 2, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat fungsi serving untuk menerima raw tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _get_transform_features_fn(model, tf_transform_output):
    """Creates the TFMA signature that transforms raw serialized examples."""
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function
    def transform_features_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        return model.tft_layer_eval(raw_features)

    return transform_features_fn


def run_fn(fn_args: FnArgs):
    """Fungsi utama yang dipanggil oleh TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=None, batch_size=32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, num_epochs=None, batch_size=32
    )

    if fn_args.hyperparameters:
        from kerastuner import HyperParameters
        hp = HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hp = None

    model = _build_keras_model(hp=hp)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_binary_accuracy", patience=3, restore_best_weights=True
            )
        ],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
        "transform_features": _get_transform_features_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
