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




    # Convolutional layers with fixed parameters
    # x = tf.keras.layers.Conv2D(
    #     filters=64,  # Number of filters
    #     kernel_size=3,  # Size of the kernel
    #     activation='relu'
    # )(input_layer)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # # x = tf.keras.layers.Conv2D(
    # #     filters=128,  # Number of filters
    # #     kernel_size=3,  # Size of the kernel
    # #     activation='relu'
    # # )(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # # x = tf.keras.layers.Conv2D(
    # #     filters=256,  # Number of filters
    # #     kernel_size=3,  # Size of the kernel
    # #     activation='relu'
    # # )(x)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # # Flatten and dense layers
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(
    #     units=128,  # Number of units in the dense layer
    #     activation='relu'
    # )(x)
    # x = tf.keras.layers.Dropout(0.3)(x)  # Dropout rate






    # # Output layer
    # output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)  # Binary classification

    # # Create the model
    # model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-1),  # Fixed learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary for debugging purposes (optional)
    model.summary()

    return model

def run_fn(fn_args: FnArgs) -> None:
    """Defines and trains the model.
    Args:
        fn_args: Holds args as name/value pairs. fn_args: This is an object of type FnArgs that holds arguments as name/value pairs. 
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    print(tf_transform_output.transformed_feature_spec())
    train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
    # for features in train_set.take(1):
    #     print(features.keys())

    val_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)

    # Build the model
    model = build_model()

    # Callbacks
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )
    
    # Train the model
    model.fit(x=train_set, validation_data=val_set, callbacks=[tensorboard_callback])

    # Save the model
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)
