import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

petrol_cons = pd.read_csv(r'J:\Documents\Datahound\Datasets\petrol-consumption\petrol_consumption.csv')
print(petrol_cons.head())

# For regression problems in general you have to normalize the feature input data
X = petrol_cons.iloc[:, 0:4].values
y = petrol_cons.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=2, epochs=100, verbose=1, validation_split=0.2)

pred_train = model.predict(X_train)
print(np.sqrt(mean_squared_error(y_train, pred_train)))

pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
