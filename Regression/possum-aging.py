import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


dataset = pd.read_csv(r'J:\Documents\Datahound\Datasets\possum-regression\possum.csv')
print(dataset.head())
print(dataset.isna().sum())
dataset.fillna(dataset.mean(), inplace=True)
print(dataset.isna().sum())

print(dataset.corr())
# sns.pairplot(dataset, diag_kind='kde', hue='age')
# plt.show()

dataset = pd.get_dummies(dataset, columns=['Pop', 'sex'], prefix='', prefix_sep='')
print(dataset.tail())

X = dataset.drop('age', axis=1)
y = dataset['age']

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
history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, epochs=500)
plot_loss(history)
print('\nNow evaluating')
model.evaluate(X_test, y_test, verbose=1)