import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)


def preprocessing(data):
    # Getting rid of outliers
    data.loc[df['wv (m/s)'] == -9999.0, 'wv (m/s)'] = 0.0
    data.loc[df['max. wv (m/s)'] == -9999.0, 'max. wv (m/s)'] = 0.0

    # Taking values every hours
    data = data[5::6]  # df[start,stop,step]

    wv = data.pop('wv (m/s)')
    max_wv = data.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = data.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    data['Wx'] = wv * np.cos(wd_rad)
    data['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    data['max Wx'] = max_wv * np.cos(wd_rad)
    data['max Wy'] = max_wv * np.sin(wd_rad)

    date_time = pd.to_datetime(data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(datetime.datetime.timestamp)

    day = 24 * 60 * 60  # Time is second within a single day
    year = 365.2425 * day  # Time in second withon a year

    data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return (data)


def split(data):
    n = data.shape[0]

    train_df = data.iloc[
               0: n * 70 // 100]  # "iloc" because we have to select the lines at the indicies 0 to int(n*0.7) compared to "loc"
    val_df = data.iloc[n * 70 // 100: n * 90 // 100]
    test_df = data.iloc[n * 90 // 100:]

    return (train_df, val_df, test_df)


df_preprocessed = preprocessing(df)
train_df, val_df, test_df = split(df_preprocessed)
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Forecasting temperature the next hour given the 48 previous features
type(train_df)
# We have to convert pandas dataframe into numpy array
# Looking at all features for the past 2 days
lookback = 48
# Trying to predict the temperature for the next day
delay = 24
window_length = lookback + delay
batch_size = 32


def create_dataset(X, y, delay=24):
    Xs, ys = [], []
    for i in range(lookback, len(X) - delay):
        # every one hour, we take the past 48 hours of features
        v = X.iloc[1 - lookback:i].to_numpy()
        Xs.append(v)
        # Every timestep we take the temperature the next delay
        w = y.iloc[i + delay]
        ys.append(w)
        return np.array(Xs), np.array(ys)


X_train, y_train = create_dataset(train_df, train_df['T (degC)'], delay=delay)
X_val, y_val = create_dataset(val_df, val_df['T (degC)'], delay=delay)

print("X_train shape {}: ".format(X_train.shape))
print("y_train shape is {}: ".format(y_train.shape))
print("\nX_val shape is {}: ".format(X_val.shape))
print("y_val shape is {}: ".format(y_val.shape))


# Naive evaluation for comparing model performance
def naive_eval_arr(X, y, lookback, delay):
    batch_maes = []
    for i in range(0, len(X)):
        # For all elements in the batch, we are saying the prediction of temp is equal to the last temp recorded within
        # 48 hours
        preds = X[i, -1, 1]
        mae = np.mean(np.abs(preds - y[i]))
        batch_maes.append(mae)
    return np.mean(batch_maes)


naive_loss_arr = naive_eval_arr(X_val, y_val, lookback=lookback, delay=delay)
# Round the value
naive_loss_arr = round(naive_eval_arr(X_val, y_val, lookback=lookback, delay=delay), 2)
# This is the acc of the naive model, the idea is to beat it with more sophisticated models
print(naive_loss_arr)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(lookback, 19)),
    tf.keras.layers.Dense(32, activation='relu'),
    # We try to predict only one value for now
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)


def plot(history, naive_loss):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validaiton loss')
    plt.title('Training and validation loss')
    plt.axhline(y=naive_loss, color='r')
    plt.legend()
    plt.show()


# Simple dense model performed better than the naive one
plot(history, naive_loss_arr)


# Generator
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=32, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        # We're not shuffling timesteps but elements within a batch, it is important to keep the data in the time order
        if shuffle == True:
            # return an array containing size elements ranging from min_index+lookback to max_index
            rows = np.random.randint(min_index + lookback, max_index - delay - 1, size=batch_size)
        else:
            # since we are incrementing on i, if its value is greater than the max_index then start from beginning
            if i + batch_size >= max_index - delay - 1:
                # we need to start from index lookback since we want to take lookback elements here
                i = min_index + lookback
                # array with indexes of each sample in the batch
                rows = np.arange(i, min(i + batch_size, max_index))
                # rows represent the number of samples in one batch
                i += len(rows)
                # shape = (batch_size, lookback, nbr_of_features)
                samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
                # shape = (batch_size,)
                targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            # from one index given by rows[j] we are picking lookback previous elements in the dataset
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            # we only want to predict temperature for now, [1] is the second column
            targets[j] = data[rows[j] + delay][1]

            yield samples, targets


data_train = train_df.to_numpy()
train_gen = generator(data=data_train, lookback=lookback, delay=delay, min_index=0, max_index=len(data_train),
                      shuffle=True, batch_size=batch_size)

data_val = val_df.to_numpy()
val_gen = generator(data=data_val, lookback=lookback, delay=delay, min_index=0, max_index=len(data_val),
                    batch_size=batch_size)

data_test = test_df.to_numpy()
test_gen = generator(data=data_val, lookback=lookback, delay=delay, min_index=0, max_index=len(data_test),
                     batch_size=batch_size)

print(next(iter(train_gen))[0].shape)
print(next(iter(train_gen))[1].shape)


# Baseline
def naive_eval_gen(generator):
    batch_maes = []
    for step in range(len(data_val) - lookback):
        samples, targets = next(generator)
        # for all elements in the batch, we're saying the prediction of temp is equal to the last temp
        # recorded within 48hrs
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)


naive_loss_gen = round(naive_eval_gen(val_gen), 2)
print(naive_loss_gen)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(lookback, 19)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')
history = model.fit(train_gen, epochs=30, steps_per_epoch=data_train.shape[0]//batch_size, validation_data=val_gen,
                    validation_steps=data_val.shape[0]//batch_size)
plot(history, naive_loss_gen)


