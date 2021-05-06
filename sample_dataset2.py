import tensorflow as tf
import pathlib
from tensorflow import keras

from repvgg import create_RepVGG_B1

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    'flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
roses = list(data_dir.glob('roses/*'))
batch_size = 16
img_height = 180
img_width = 180
num_classes = 5

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model = keras.applications.VGG16(include_top=True, weights=None, classes=5, input_shape=[180, 180, 3])

preprocess = keras.layers.experimental.preprocessing.Rescaling(
    1./255, input_shape=(img_height, img_width, 3))
run_model = create_RepVGG_B1(num_classes=num_classes)
model = keras.Sequential([preprocess, run_model])

# model = keras.Sequential([
#   keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Flatten(),
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dense(num_classes)
# ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

