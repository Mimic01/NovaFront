import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
print(dataset.isna().sum())
dataset.fillna(dataset.mean(), inplace=True)
print(dataset.isna().sum())

# sns.pairplot(dataset, diag_kind='kde')
# plt.show()

dataset = pd.get_dummies(dataset, columns=['Pop', 'sex'], prefix='', prefix_sep='')
print(dataset.head())

X = dataset.drop(['f', 'm'], axis=1)
print(X.head())
sex_cols = ['f', 'm']
y = dataset[sex_cols]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1, validation_split=0.2)

results = model.evaluate(X_test, y_test, verbose=1)
print("test loss, test acc:", results)

print("Generate predictions for 3 samples")
predictions = model.predict(X_test[:3])
print("predictions shape:", predictions.shape)

plot_loss(history)