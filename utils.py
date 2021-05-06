import tensorflow as tf 


def process_image_and_label(image, label):
    # image = tf.cast(image, dtype=tf.float32) / 255.0
    image = tf.cast(image, dtype=tf.float32)
    label = tf.squeeze(label, axis=1)
    label = tf.one_hot(label, 100, off_value=0)
    return image, label