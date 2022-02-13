import keras.models
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import pathlib
import os, shutil
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import optimizers, regularizers
import math

# Achieving acc: 0.9824 on the YIKES! Spiders -15 Species Classification Dataset
# https://www.kaggle.com/gpiosenka/yikes-spiders-15-species/code

base_url = 'J:\Documents\Datahound\Datasets\spiders-dataset'
dataset_csv_urls_dir = 'J:\Documents\Datahound\Datasets\spiders-dataset\spiders.csv'
train_url = os.path.join(base_url, 'train')
test_url = os.path.join(base_url, 'test')
valid_url = os.path.join(base_url, 'valid')

# img = Image.open(os.path.join(base_url, csv_urls['filepaths'].iloc[200]))
# img.show()

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_url, target_size=(224, 224), color_mode='rgb',
                                                    classes=None, class_mode='categorical', batch_size=32)
test_generator = test_datagen.flow_from_directory(test_url, target_size=(224, 224), color_mode='rgb',
                                                  classes=None, class_mode='categorical', batch_size=32)
valid_generator = valid_datagen.flow_from_directory(valid_url, target_size=(224, 224), color_mode='rgb',
                                                    classes=None, class_mode='categorical', batch_size=32)

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
    keras.layers.Dense(15, activation='softmax')
])
print(model.summary())

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
checkpoint_url = 'J:/NovaFront/ImageClassification/Spiders'
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_url, save_weights_only=False,
                                                         monitor='val_loss', save_best_only=True)

model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
history = model.fit(train_generator, epochs=30, validation_data=valid_generator,
                    callbacks=[early_stopping_cb, model_checkpoint_cb])

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

model.evaluate(test_generator)


s = 20 * 2185 // 32
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)

base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3),
                                                     pooling='max')
x = base_model.output
x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = keras.layers.Dense(256, kernel_regularizer= regularizers.l2(l = 0.016), activity_regularizer= regularizers.l1(0.006),
                       bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = keras.layers.Dropout(rate=.45, seed=123)(x)
output = keras.layers.Dense(15, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(keras.optimizers.SGD(learning_rate), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(train_generator, epochs=15, validation_data=valid_generator,
                    callbacks=[early_stopping_cb, model_checkpoint_cb])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

acc_loss_plotter(acc, loss, val_acc, val_loss)

chkpt = 1
