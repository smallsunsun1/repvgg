from tensorflow import keras
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

l1 = keras.layers.Conv2D(64, (3, 3))
b = l1(tf.zeros(shape=[1, 100, 100, 32]))
print(tf.shape(l1.get_weights()[1]))
l1.set_weights([tf.zeros(shape=[3, 3, 32, 64]), tf.zeros(shape=[64])])
