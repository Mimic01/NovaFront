import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import plotly.graph_objects as go

data = pd.read_csv(r'J:\Documents\Datahound\Datasets\DelhiClimate\DailyDelhiClimateTest.csv')
print(data.head())
print(data.shape)

data = data.interpolate(method='linear', axis=0)
print(data.isnull().sum())

# plt.plot(data.date, data.meantemp)
# plt.show()

multivariate_df = data[['meanpressure', 'humidity', 'wind_speed', 'meantemp']]
print(multivariate_df.head())


def normalizer_df(data):
    scaler = MinMaxScaler().fit(data.values)
    data_normd = scaler.transform(data.values)
    data = pd.DataFrame(data_normd, index=data.index, columns=data.columns)
    return data


def split_sequence(sequence, n_steps, multivariate=False):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        if multivariate:
            seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix - 1, -1]
        else:
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


multi_normed = normalizer_df(multivariate_df)
raw_sequence = multi_normed.values
n_steps = 3
n_features = 3

X, y = split_sequence(raw_sequence, n_steps, True)
for i in range(len(X)):
    print(X[i], y[i])
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)

EPOCHS = 200


def display_training_curves(training, validation, yaxis):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS + 1), mode='lines+markers', y=training, marker=dict(color="dodgerblue"),
                   name='Train'))
    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS + 1), mode='lines+markers', y=validation, marker=dict(color='darkorange'),
                   name="Val"))
    fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs", template="plotly_white")
    fig.show()

batch_size = 4
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(150, activation='tanh', return_sequences=False, input_shape=(n_steps, n_features),
                         kernel_regularizer='l2', recurrent_regularizer='l2', dropout=0.2, recurrent_dropout=0.3),
    # tf.keras.layers.LSTM(100, activation='tanh', kernel_regularizer='l2', recurrent_regularizer='l2', dropout=0.2,
    #                      recurrent_dropout=0.3, return_sequences=True),
    # tf.keras.layers.LSTM(50, activation='tanh', kernel_regularizer='l2', recurrent_regularizer='l2', dropout=0.2,
    #                      recurrent_dropout=0.4),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1)
])
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000,
                                                               decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(X, y, epochs=EPOCHS, validation_data=(X_valid, y_valid), verbose=1)
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')
