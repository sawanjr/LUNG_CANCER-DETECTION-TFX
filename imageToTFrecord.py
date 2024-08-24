import os
import tensorflow as tf
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(image_dir, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    
    for label, class_name in enumerate(os.listdir(image_dir)):
        class_path = os.path.join(image_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            with open(image_path, 'rb') as img_file:
                img = img_file.read()
                
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(img),
                'label': _int64_feature(label)
            }))
            writer.write(example.SerializeToString())
    
    writer.close()
# Example usage
image_dir = 'Data/Train/'
output_file = 'tfrecordImagesData/images.tfrecord'
create_tfrecord(image_dir, output_file)
