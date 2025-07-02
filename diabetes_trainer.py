"""Training module
"""
 
import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft 
from diabetes_transform import (
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)
 
def get_model(show_summary=True):
    """Mendefinisikan arsitektur model Keras dengan normalisasi dan regularisasi.

    Args:
        show_summary (bool): Jika True, akan menampilkan summary model.

    Returns:
        tf.keras.Model: Model Keras siap training.
    """
    input_features = []

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = tf.keras.layers.concatenate(input_features)

    x = tf.keras.layers.BatchNormalization()(concatenate)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    if show_summary:
        model.summary()

    return model
 
def gzip_reader_fn(filenames):
    """Memuat file TFRecord terkompresi GZIP.

    Args:
        filenames (list): Daftar path file TFRecord.

    Returns:
        tf.data.TFRecordDataset: Dataset TFRecord.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
 
def get_serve_tf_examples_fn(model, tf_transform_output):
    """Mengembalikan fungsi untuk serving signature model TensorFlow.

    Args:
        model (tf.keras.Model): Model Keras terlatih.
        tf_transform_output (TFTransformOutput): Output transformasi TFX.

    Returns:
        Callable: Fungsi serving TensorFlow.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()
 
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Mengembalikan output untuk digunakan sebagai serving signature model TFX."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
 
        transformed_features = model.tft_layer(parsed_features)
 
        outputs = model(transformed_features)
        return {"outputs": outputs}
 
    return serve_tf_examples_fn
 
def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Membuat dataset (features, labels) untuk training/tuning model.

    Args:
        file_pattern (str): Pola file TFRecord.
        tf_transform_output (TFTransformOutput): Output transformasi TFX.
        batch_size (int): Ukuran batch.

    Returns:
        tf.data.Dataset: Dataset untuk training atau evaluasi.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
 
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
 
    return dataset
 
def run_fn(fn_args):
    """Melatih model berdasarkan argumen dari Trainer TFX.

    Args:
        fn_args: Objek FnArgs berisi argumen konfigurasi training.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
 
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)
 
    model = get_model()
 
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
 
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=50
    )
 
    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )
    
    plot_model(
        model, 
        to_file='images/model_plot.png', 
        show_shapes=True, 
        show_layer_names=True
    )