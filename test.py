from tensorflow import keras
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from repvgg import get_RepVGG_func_by_name, repvgg_model_convert, create_RepVGG_A0

repvgg_module = create_RepVGG_A0(num_classes=100)

inputs = tf.ones(shape=[1, 224, 224, 3], dtype=tf.float32)
res = repvgg_module(inputs)
repvgg_model_convert(repvgg_module, "./saved_model")

model = tf.saved_model.load("./saved_model")
res2 = model(inputs)

print(tf.reduce_max(tf.abs(res2 - res)))
