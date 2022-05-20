import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Determine whether the signal shows the presence of some object or just empty air.
ion = pd.read_csv(r"J:\Documents\Datahound\Datasets\ionosphere\ionosphere.csv", index_col=0)
ion.columns = ["V" + str(n) for n in range(ion.shape[1])]
print(ion.head())

df = ion.copy()
df['Class'] = df['V33'].map({'g': 0, 'b': 1})
df = df.drop(['V33', 'V0'], axis=1)
print(df.head())

X = df.drop(['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001,
                                               restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=512, epochs=1000,
                    callbacks=[early_stopping], verbose=1)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)
