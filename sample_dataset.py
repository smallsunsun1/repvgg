import tensorflow as tf 
import numpy as np 
import tensorflow_addons as tfa
from tensorflow import keras
from repvgg import create_RepVGG_B1

from utils import process_image_and_label

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
# model = create_RepVGG_B1(num_classes=100)
model = keras.applications.VGG16(include_top=True, weights=None, classes=100, input_shape=[32, 32, 3])

train_image_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_train))
train_label_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_train))
test_image_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_test))
test_label_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_test))


train_dataset = tf.data.Dataset.zip((train_image_data, train_label_data))
train_dataset = train_dataset.batch(4)
train_dataset = train_dataset.map(process_image_and_label, tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

step = tf.Variable(0, trainable=False)
lr_fn = keras.optimizers.schedules.PiecewiseConstantDecay([100, 10000], [1e-4, 1e-3, 1e-4])
lr = lr_fn(step)
optimizer = keras.optimizers.SGD(lr)
loss_obj = keras.losses.CategoricalCrossentropy(from_logits=True)

@tf.function
def run_one_step(inputs, label):
    with tf.GradientTape() as tape:
        output = model(inputs)
        loss = tf.reduce_mean(loss_obj(label, output))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    step.assign_add(1)
    if tf.math.equal(tf.math.mod(step, 100), 0):
        tf.print(loss)
    

for idx, elem in enumerate(train_dataset):
    run_one_step(elem[0], elem[1])
    # print(tf.shape(elem[0]))
    # tf.io.write_file("./test_{}.png".format(idx), tf.image.encode_png(elem[0][0])) 
    # if idx == 1:
    #     break