import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error/Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


train_path = "J:/Documents/Datahound/Datasets/mercedes-benz-greener-manufacturing/train.csv"
test_path = "J:/Documents/Datahound/Datasets/mercedes-benz-greener-manufacturing/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(train_df.shape)
print(test_df.shape)
print(train_df.head())
print(test_df.head())
print(train_df.isna().sum())

cols = train_df.columns
num_cols = train_df._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

train_df.rename({'y': 'label'}, axis=1, inplace=True)
print(train_df.head())
train_df = pd.get_dummies(train_df, columns=cat_cols, prefix='', prefix_sep='')
print(train_df.tail())

X = train_df.drop('label', axis=1)
y = train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(X_train)

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
print(model.summary())
history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, epochs=30)
plot_loss(history)
print("\nEvaluating")
model.evaluate(X_test, y_test, verbose=1)


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
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, epochs=500,
                    callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50))
plot_loss(history)
stopval = 5