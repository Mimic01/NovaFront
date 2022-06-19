import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import plotly.graph_objects as go

data = pd.read_csv(r"J:\Documents\Datahound\Datasets\USPopulation\POP.csv")
print(data.head(200))
print(data.shape)

# plt.plot(data.date, data.value)
# plt.show()

df = data[['date', 'value']]
print(df.head())


def normalizer_df(data):
    scaler = MinMaxScaler().fit(data.values.reshape(-1, 1))
    data_normd = scaler.transform(data.values.reshape(-1, 1))
    data = pd.DataFrame(data_normd, index=data.index)
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


uni_normed = normalizer_df(df['value'])
raw_sequence = uni_normed.values.tolist()
n_steps = 3
n_features = 1

X, y = split_sequence(raw_sequence, n_steps)
for i in range(len(X)):
    print(X[i], y[i])
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)

EPOCHS = 200
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)),
    tf.keras.layers.LSTM(5, activation='tanh'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=EPOCHS, validation_data=(X_valid, y_valid), verbose=1)


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


display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')