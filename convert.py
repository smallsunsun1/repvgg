import tensorflow as tf 
from tensorflow import keras
from repvgg import RepVGGBlock

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = RepVGGBlock(3, 64, 1)
inputs = tf.random.uniform(shape=[1, 224, 224, 3], dtype=tf.float32)
o1 = model(inputs, training=False)
model.switch_to_deploy()
o2 = model(inputs, training=False)
print(tf.reduce_mean(o1 - o2))