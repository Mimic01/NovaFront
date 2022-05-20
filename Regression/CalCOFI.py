import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras import backend as K

# Is there a relationship between water salinity & water temperature?
# Can you predict the water temperature based on salinity?

dataPath = "J:/Documents/Datahound/Datasets/CalCOFI/bottle.csv"
raw_dataset = pd.read_csv(dataPath)
dataset = raw_dataset.copy()

print(dataset.shape)
print(dataset.tail())


# for column in dataset.columns:
#     n_miss = dataset[[column]].isnull().sum()
#     perc = n_miss / dataset.shape[0] * 100
#     print('> %s, Missing %s (%.1f%%)' % (column, n_miss, perc))
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


dataset = dataset[['T_degC', 'Salnty']]
dataset.columns = ['Temperature', 'Salinity']
print(dataset.head())

# sns.pairplot(dataset, kind="reg")
# plt.show()
print(dataset.isnull().sum())
dataset.fillna(dataset.mean(), inplace=True)

X = np.array(dataset['Salinity']).reshape(-1, 1)
y = np.array(dataset['Temperature']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

normalizer = tf.keras.layers.Normalization(input_shape=[1, ], axis=None)
normalizer.adapt(np.array(X_train))
print(normalizer.mean.numpy())

singleInput_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
singleInput_model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.Adam(learning_rate=0.001),
                          metrics=["mean_squared_error"])
singleInput_model.build(X_train.shape)
print(singleInput_model.summary())
history = singleInput_model.fit(X_train, y_train, epochs=30)
plot_loss(history)

test_results = {}
# test_results['singleInput_model'] = singleInput_model.evaluate()
