"""Tuner module
"""

from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
import kerastuner as kt
from kerastuner.engine import base_tuner
import tensorflow as tf
import tensorflow_transform as tft
from diabetes_transform import (
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

TunerFnResult = NamedTuple(
    'TunerFnResult',
    [('tuner', base_tuner.BaseTuner), ('fit_kwargs', Dict[Text, Any])]
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def gzip_reader_fn(filenames):
    """Memuat data TFRecord yang dikompresi menggunakan GZIP.

    Args:
        filenames: List path file TFRecord.

    Returns:
        tf.data.TFRecordDataset dengan kompresi GZIP.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Membuat dataset TensorFlow untuk TFX tuner dan trainer.

    Args:
        file_pattern: Pola file TFRecord input.
        tf_transform_output: Output transformasi dari TFX Transform.
        batch_size: Ukuran batch dataset.

    Returns:
        tf.data.Dataset siap untuk training atau evaluasi.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset

def model_builder(hp):
    """Membangun model Keras dengan hyperparameter yang dapat dituning.

    Args:
        hp: Objek HyperParameters dari KerasTuner.

    Returns:
        tf.keras.Model yang telah dikompilasi dan siap dilatih.
    """
    hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    hp_units1 = hp.Int('units_layer1', min_value=64, max_value=256, step=32)
    hp_units2 = hp.Int('units_layer2', min_value=32, max_value=128, step=16)
    hp_units3 = hp.Int('units_layer3', min_value=16, max_value=64, step=16)

    hp_dropout = hp.Choice('dropout_rate', [0.2, 0.3, 0.4])
    hp_l2 = hp.Choice('l2_reg', [1e-3, 1e-4])

    input_features = [
        tf.keras.Input(shape=(1,), name=transformed_name(feature))
        for feature in NUMERICAL_FEATURES
    ]

    concat = tf.keras.layers.concatenate(input_features)

    x = tf.keras.layers.BatchNormalization()(concat)

    x = tf.keras.layers.Dense(
        hp_units1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(hp_l2)
    )(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)

    x = tf.keras.layers.Dense(
        hp_units2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(hp_l2)
    )(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)

    x = tf.keras.layers.Dense(
        hp_units3,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(hp_l2)
    )(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi tuner TFX menggunakan KerasTuner Hyperband untuk pencarian hyperparameter.

    Args:
        fn_args: Objek FnArgs dari TFX yang berisi path data dan informasi training.

    Returns:
        TunerFnResult yang berisi:
            - tuner: Objek KerasTuner Hyperband.
            - fit_kwargs: Argument untuk fungsi fit().
    """
    tuner = kt.Hyperband(
        model_builder,
        objective='val_binary_accuracy',
        max_epochs=50,
        factor=8,
        directory=fn_args.working_dir,
        project_name='kt_hyperband'
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        }
    )
