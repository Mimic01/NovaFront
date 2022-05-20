import keras.callbacks
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import regularizers


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


dataset = pd.read_csv(r"J:\Documents\Datahound\Datasets\orangevsgrapefruit\citrus.csv")
print(dataset.head())
print(dataset.isna().sum())
dataset.fillna(dataset.mean(), inplace=True)
print(dataset.isna().sum())

dataset = pd.get_dummies(dataset, columns=['name'], prefix='', prefix_sep='')
print(dataset.head())

X = dataset.drop(['grapefruit', 'orange'], axis=1)
print(X.head())
fruitName_cols = ['grapefruit', 'orange']
y = dataset[fruitName_cols]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(X_train)

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid')
])
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=200, verbose=1)]
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96,
                                                          staircase=True)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
history = model.fit(X_train, y_train, epochs=500, batch_size=64, callbacks=callbacks, validation_split=0.2, verbose=1)
plot_loss(history)
stopvar = 5