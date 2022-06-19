import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
fname = os.path.splitext(zip_path)

f = open(fname[0])
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
# you get 14 columns which are the 14 weather features
print(header)
# each line is a timestep: a record of a date and 14 weather-related values
print(len(lines))
# parse data into numpy array
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# temperature plot over the whole temporal range of the dataset
temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)

# temperature plot over the first 10 days of the dataset
plt.plot(range(1440), temp[:1440])

# Problem: Given data going as far back as 'lookback' timesteps (a timestep is 10mins) and samples every 'steps'
# timesteps, can you predict the temperature in 'delay' timesteps?
# lookback = 720 -- observations will go back 5 days
# steps = 6 -- observations will be samples at one data point per hour
# delay = 144 --targets will be 24 hours in the future

# Preprocess the data, normalizing. We'll use 200,000 timesteps as training data.
# we'll substract the mean of each time series and divide it by standard deviation
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True,
                      step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step,
                    batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step,
                     batch_size=batch_size)

# how many steps to draw from val_gen in order to see the entire validation set
val_steps = (300000 - 200001 - lookback)
# how many steps to drawe fom test_gen in order to see the entire test set
test_steps = (len(float_data) - 300001 - lookback)


# baseline
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)


# this gives a mae of 0.29 and it can be better understood by 0.29 * temperature_std = 2.57C
evaluate_naive_method()
celsius_mae = 0.29 * std[1]

# Simple atemporal model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)


def plot_model_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


plot_model_loss(history)

# GRU based model
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, input_shape=(None, float_data.shape[-1])),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(), los='mae')
history = model.fit(train_gen, steps_per_epoch=500, validation_data=val_gen, validation_steps=val_steps)
plot_model_loss(history)

# Dropout-regularized GRU-based model
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optmizers.RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)
plot_model_loss(history)

# Stacked GRU model
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True,
                        input_shape=(None, float_data.shape[-1])),
    tf.keras.layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)
