import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers

(ds_test, ds_train, ds_valid), ds_info = tfds.load('oxford_flowers102', split=['test', 'train', 'validation'],
                                                   shuffle_files=True,
                                                   data_dir="J:/Documents/Datahound/Datasets/oxford_flowers102",
                                                   as_supervised=True,
                                                   with_info=True)
print(ds_info.features)


def normalize_img(image, label):
    return tf.cast(image, tf.float32), label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.padded_batch(batch_size=128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_test = ds_train.take(100)
# ds_test = ds_test.cache()
# ds_test = ds_train.unbatch()
# ds_test = ds_test.batch(128)
# ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(rate=.2, seed=123),
    keras.layers.Dense(102, activation='softmax')
])
print(model.summary())

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
checkpoint_url = 'J/NovaFront/ImageClassification/OxfordFlowers'
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_url, save_weights_only=False, monitor='val_loss',
                                                         save_best_only=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
history = model.fit(ds_train, epochs=30, callbacks=[early_stopping_cb, model_checkpoint_cb])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


def acc_loss_plotter(acc, loss, val_acc, val_loss):
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


acc_loss_plotter(acc, loss, val_acc, val_loss)

# model.evaluate(ds_test)
