import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import plotly.graph_objects as go

data = pd.read_csv(r"J:\Documents\Datahound\Datasets\brazilian-cities-temp-forecast\station_belem.csv")
print(data.head())
print(data.shape)

invalid_value = data.iloc[4]['JAN']
data.replace(invalid_value, np.NaN, inplace=True)
data = data.interpolate(method='linear')
print(data.head())

# plt.plot(data.YEAR, data.JAN)
# plt.show()

univariate_df = data[['YEAR', 'JAN']]
print(univariate_df.head())


def normalizer_df(data):
    scaler = MinMaxScaler().fit(data.values)
    data_normd = scaler.transform(data.values)
    data = pd.DataFrame(data_normd, index=data.index, columns=data.columns)
    return data


uni_normed = normalizer_df(univariate_df)


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


raw_sequence = uni_normed['JAN'].values.tolist()
n_steps = 3
n_features = 1

X, y = split_sequence(raw_sequence, n_steps)
X = X.reshape((X.shape[0], X.shape[1], n_features))
for i in range(len(X)):
    print(X[i], y[i])
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)

EPOCHS = 200
# model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)),
#     tf.keras.layers.LSTM(50, activation='tanh'),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# history = model.fit(X, y, epochs=EPOCHS, validation_data=(X_valid, y_valid), verbose=1)


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
        go.Scatter(x=np.arange(1, EPOCHS+1), mode='lines+markers', y=validation, marker=dict(color='darkorange'),
                   name="Val"))
    fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs", template="plotly_white")
    fig.show()


# display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')

# Multivariate
multi_columns = ['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
multivariate_df = data[multi_columns]
print(multivariate_df.head())

multi_normed = normalizer_df(multivariate_df)
raw_sequence = multi_normed.loc[:, 'JAN':'DEC'].values
n_steps = 3
# 11 months as features, 1 month (december) as label
n_features = 11

X, y = split_sequence(raw_sequence, n_steps, True)
for i in range(len(X)):
    print(X[i], y[i])
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(150, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features),
                         kernel_regularizer='l2', recurrent_regularizer='l2', dropout=0.2, recurrent_dropout=0.3),
    tf.keras.layers.LSTM(100, activation='tanh', kernel_regularizer='l2', recurrent_regularizer='l2', dropout=0.2,
                         recurrent_dropout=0.3, return_sequences=True),
    tf.keras.layers.LSTM(50, activation='tanh', kernel_regularizer='l2', recurrent_regularizer='l2', dropout=0.2,
                         recurrent_dropout=0.4),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1)
])
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000,
                                                               decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(X, y, epochs=EPOCHS, validation_data=(X_valid, y_valid), verbose=1)
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')