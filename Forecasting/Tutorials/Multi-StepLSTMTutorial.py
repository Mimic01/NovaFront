import numpy as np
import pandas as pd
import tensorflow as tf


# A time series forecasting problem that requires a prediction of multiple steps into the future can be referred to as
# multi-step time series forecasting
# 2 types of LSTM models can be used for multi step: Vector output model and encoder-decoder model

# For example, for:
# [10, 20, 30, 40, 50, 60, 70, 80, 90]
# input: [10, 20, 30]
# output: [40, 50]

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
for i in range(len(X)):
    print(X[i], y[i])

# LSTM expects data to have a three-dimensional structure of [samples, timesteps, features],
# and in this case, we only have one feature so the reshape is straightforward.
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Vector Output model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)),
    tf.keras.layers.LSTM(100, activation='relu'),
    tf.keras.layers.Dense(n_steps_out)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=1)
# As expected by the model, the shape of the single sample of input data when making the prediction must be
# [1, 3, 1] for the 1 sample, 3 time steps of the input, and the single feature.
x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

# Encoder-Decoder Model
# Developer for variable length output sequences, for input and output sequences problems (sequence-to-sequence)
# like translating text. It can also be used for multi-step time series forecasting

# The encoder is a model responsible for reading and interpreting the input sequence.
# The output of the encoder is a fixed length vector that represents the model’s interpretation of the sequence.

X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
n_features = 1
# 3-dim input [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], n_features)
# must also be 3-dim in the case of encoder-decoder model because the model will predict a given number of time-steps
# with a given number of features for each input sample
y = y.reshape(y.shape[0], y.shape[1], n_features)

model = tf.keras.models.Sequential([
    # The input layer is the encoder which is responsible for reading and interpreting the input sequence, the output of
    # the encoder is a fized length vector that represents the model's interpretation of the sequence. The encoder is
    # usually a vanilla LSTM
    tf.keras.layers.LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)),
    # First the fixed-length output of the encoder is repeated, once for each required time step in the output sequence
    tf.keras.layers.RepeatVector(n_steps_out),
    # This sequence is then provided to the decoder layer, the layer must outpout a value for each value in the output
    # time step
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
    # We can use the same output layer or layers to make each one-step prediction in the output sequence. This can be
    # achieved by wrapping the output time step in a TimeDistributed layer, which can be interpreted
    # by a single output model
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=1)


x_input = np.array([70, 80, 90])
x_input = x_input.reshape(1, n_steps_in, n_features)
yhat = model.predict(x_input, verbose=0)
print(yhat)

