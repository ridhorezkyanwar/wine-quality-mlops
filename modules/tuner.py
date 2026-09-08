"""Tuner module untuk hyperparameter tuning Wine Quality pipeline."""

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

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

NUM_TRIALS = 5


def transformed_name(key):
    return key + "_xf"


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
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


def _build_tunable_model(hp):
    units = hp.Int("units", min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

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


def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=10, batch_size=32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, num_epochs=1, batch_size=32
    )

    tuner = kt.RandomSearch(
        _build_tunable_model,
        objective=kt.Objective("val_binary_accuracy", direction="max"),
        max_trials=NUM_TRIALS,
        directory=fn_args.working_dir,
        project_name="wine_quality_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
