"""Pipeline TFX untuk Wine Quality Classification."""

import os
from absl import logging
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.components import Tuner

# ── Path konfigurasi ──────────────────────────────────────────────────────────
PIPELINE_NAME = "ridhorezkyanwar-pipeline"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PIPELINE_ROOT = os.path.join(BASE_DIR, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata", "metadata.db")
SERVING_MODEL_DIR = os.path.join(BASE_DIR, "serving_model")

TRANSFORM_MODULE = os.path.join(BASE_DIR, "modules", "transform.py")
TRAINER_MODULE = os.path.join(BASE_DIR, "modules", "trainer.py")
TUNER_MODULE = os.path.join(BASE_DIR, "modules", "tuner.py")

TRAIN_STEPS = 100
EVAL_STEPS = 50


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    transform_module: str,
    trainer_module: str,
    tuner_module: str,
    serving_model_dir: str,
    metadata_path: str,
):
    """Membuat TFX pipeline untuk Wine Quality Classification."""

    # 1. ExampleGen – membaca CSV dan split train/eval
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_root, output_config=output_config)

    # 2. StatisticsGen – menghitung statistik dataset
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # 3. SchemaGen – membuat schema dari statistik
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # 4. ExampleValidator – validasi data terhadap schema
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # 5. Transform – preprocessing fitur
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module,
    )

    # 6. Tuner – hyperparameter tuning
    tuner = Tuner(
        module_file=tuner_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_STEPS),
    )

    # 7. Trainer – melatih model
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_STEPS),
    )

    # 8. Resolver – mendapatkan model terbaik yang sudah di-bless
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # 9. Evaluator – evaluasi model baru vs model lama
    import tensorflow_model_analysis as tfma

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key="quality_xf",
                preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.7}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -0.01},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # 10. Pusher – push model ke serving directory
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        enable_cache=True,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    BeamDagRunner().run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_DIR,
            transform_module=TRANSFORM_MODULE,
            trainer_module=TRAINER_MODULE,
            tuner_module=TUNER_MODULE,
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_path=METADATA_PATH,
        )
    )
