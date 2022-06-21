import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])
print(abalone_train.head())

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
print(abalone_features)

abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])
abalone_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
abalone_model.fit(abalone_features, abalone_labels, epochs=10)

# Now basic preprocessing and normalization of numeric columns
normalize = layers.Normalization()
normalize.adapt(abalone_features)
norm_abalone_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1),
])
norm_abalone_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)



