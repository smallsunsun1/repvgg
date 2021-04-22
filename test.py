from tensorflow import keras
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from repvgg import get_RepVGG_func_by_name, repvgg_model_convert

repvgg_module = get_RepVGG_func_by_name("RepVGG-B1")()

inputs = tf.ones(shape=[1, 224, 224, 3], dtype=tf.float32)
res = repvgg_module(inputs)
repvgg_model_convert(repvgg_module, "./saved_model")

model = tf.saved_model.load("./saved_model")
res2 = model(inputs)

print(tf.reduce_sum(res2 - res))
