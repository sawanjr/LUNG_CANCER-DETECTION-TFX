import tensorflow as tf
import tensorflow_transform as tft

# Keys
_LABEL_KEY = 'label'
_IMAGE_KEY = 'image'

def _transformed_name(key):
    return key + '_xf'

def _image_parser(image_str):
    '''Converts the images to a float tensor, resizes, and scales them.'''
    image = tf.image.decode_jpeg(image_str, channels=3)
    image = tf.image.resize(image, [64, 64])  # Resize images to 224x224
    image = tf.cast(image, tf.float32)
    return image

def _label_parser(label_id):
    '''Converts labels to one-hot encoding.'''
    # Assuming you have two classes: benign (0) and malignant (1)
    
    return label_id

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    
    # Convert the raw image and labels to a float array and
    # one-hot encoded labels, respectively.
    outputs = {
        _transformed_name(_IMAGE_KEY):
            tf.map_fn(
                _image_parser,
                tf.squeeze(inputs[_IMAGE_KEY], axis=1),
                dtype=tf.float32),
        _transformed_name(_LABEL_KEY):
            tf.map_fn(
                _label_parser,
                tf.squeeze(inputs[_LABEL_KEY], axis=1),
                dtype=tf.int64)
    }
    
    # Scale the pixel values from 0 to 1
    outputs[_transformed_name(_IMAGE_KEY)] = outputs[_transformed_name(_IMAGE_KEY)] / 255.0
    
    return outputs


#######################################################################

import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers, models
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow import keras
from kerastuner.engine import base_tuner
from typing import NamedTuple, Dict, Text, Any
from pathlib import Path

# Constants
_IMAGE_KEY = 'image_xf'
_LABEL_KEY = 'label_xf'

# Defining a structure(NamedTuple) that can hold both the tuner tool (base_tuner) and the training instructions(fit_kwargs) together
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# 1. Load compressed dataset
def _gzip_reader_fn(filenames):  
    '''Load compressed dataset
    Args:
        filenames: filenames of TFRecords to load
    Returns:
        TFRecordDataset loaded from the filenames
    '''
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=32) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records
    Args:
        file_pattern: List of files or patterns of file paths containing Example records.
        tf_transform_output: The transform output graph from TensorFlow Transform (TF Transform) that contains information about how features have been transformed or preprocessed.
        num_epochs: Integer specifying the number of times to read through the dataset. 
                    If None, cycles through the dataset forever.
        batch_size: An int representing the number of records to combine in a single batch.
    Returns:
        A dataset of dict elements, (or a tuple of dict elements and label). 
        Each dict maps feature keys to Tensor or SparseTensor objects.
    '''
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_LABEL_KEY
    )
    return dataset

# 3. Applying the preprocessing graph to model inputs
def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop("label")
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_example, feature_spec)
        transformed_features = model.tft_layer_eval(parsed_features)
        return transformed_features

    return transform_features_fn

def export_serving_model(tf_transform_output, model, output_dir):
    """Exports a keras model for serving.
    Args:
        tf_transform_output: Wrapper around output of tf.Transform.
        model: A keras model to export for serving.
        output_dir: A directory where the model will be exported to.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }

    model.save(output_dir, save_format='tf', signatures=signatures)

import tensorflow as tf
from kerastuner import HyperParameters

import tensorflow as tf

def build_model():
    """Builds a Keras CNN model for image classification without hyperparameter tuning.

    Returns:
        A compiled Keras model.
    """
    input_layer = tf.keras.Input(shape=(64,64, 3), name=_IMAGE_KEY)

    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=input_layer
    )
    base_model.trainable = False  # Freeze the base model layers

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # For binary classification, use 1 unit with sigmoid activation
    output_layer = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
