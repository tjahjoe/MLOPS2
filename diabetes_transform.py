"""Transform module
"""
 
import tensorflow as tf
import tensorflow_transform as tft
 
NUMERICAL_FEATURES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]
LABEL_KEY = 'Outcome'
 
def transformed_name(key):
    """Menambahkan suffix '_xf' pada nama fitur untuk menandai fitur yang sudah ditransformasi.

    Args:
        key: Nama fitur asli.

    Returns:
        Nama fitur dengan suffix '_xf'.
    """
    return key + "_xf"
 
 
def preprocessing_fn(inputs):
    """Fungsi preprocessing untuk mentransformasi fitur input agar siap digunakan model.

    Args:
        inputs: Dictionary yang memetakan nama fitur ke tensor fitur mentah.

    Returns:
        Dictionary yang memetakan nama fitur yang telah ditransformasi ke tensor yang telah ditransformasi.
    """
    outputs = {}
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])
    
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs