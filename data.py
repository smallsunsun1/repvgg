import os
import glob
import tensorflow as tf 

def load_image(filename: tf.Tensor):
    data = tf.io.read_file(filename)
    image_data = tf.image.decode_image(data)
    return image_data

def parse_tfrecord(input_record: tf.Tensor):
    feature_map = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=""),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        "height": tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=0),
        "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }
    record_feature = tf.io.parse_single_example(input_record, feature_map)
    record_feature["image"] = tf.io.decode_image(record_feature["image"])
    return record_feature

class ImageDataset:
    def __init__(self, image_folder):
        self.image_folder = image_folder
    def input_fn(self, is_training=True):
        filenames = os.listdir(self.image_folder)
        filenames = [os.path.join(self.image_folder, filename) for filename in filenames]
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_training:
            dataset = dataset.repeat(-1)
            dataset = dataset.shuffle(1000)
        return dataset

class TFRecordDataset:
    def __init__(self, filenames):
        self.filenames = filenames
    def input_fn(self, is_training=True):
        dataset = tf.data.TFRecordDataset(self.filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_training:
            dataset = dataset.repeat(-1)
            dataset = dataset.shuffle(1000)
        return dataset