

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
