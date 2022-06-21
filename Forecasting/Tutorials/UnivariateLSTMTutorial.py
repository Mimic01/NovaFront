import numpy as np
import pandas as pd
import tensorflow as tf


# Consider a sequence:
# [10, 20, 30, 40, 50, 60, 70, 80, 90]

# We can divide the sequence into multiple input/output patterns called samples, where three time steps are used
# as input and one time step is used as output for the one-step prediction that is being learned.
# X,				y
# 10, 20, 30		40
# 20, 30, 40		50
# 30, 40, 50		60
# ...

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps = 3
X, y = split_sequence(raw_seq, n_steps)
for i in range(len(X)):
    print(X[i], y[i])

# input shape in this case is what the model expects as input for each sample in terms of the number os time steps
# and the number of features
# we're working on univariate series, so the number of features is one, for one variable.
# almost always we'll have multiple samples, therefore the model will expect the input component of training data to
# have the dimensions or shape [samples, timesteps, features]
# so we have to reshape it to have an additional dimensions for the one feature
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# Vanilla LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=1)
# model expetcs the input shape to be three dimensional with [samples, timesteps, features] therefore, we must reshape
# the single input sample before making the prediction
x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

# Stacked LSTM
# An LSTM layer requires a 3-dimensional input and LSTMs by default will produce a two-dimensional output as an
# interpretation from the end of the sequence
# You can address this by having the LSTM output a value each time step in the input data by setting the
# return_sequences=True argument on the layer. This allows us to have 3D output from hidden LSTM layer as input to
# the text
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=1)
x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=1)
print(yhat)

# CNN-LSTM
# CNN can be very effective at extracting and learning features from one-dimensional sequence data such as univariate
# time series data
# CNN can be used in a hybrid model with an LSTM backend where the CNN is used to interpret subsequences of input that
# together are provided as a sequence to an LSTM model to interpret

# First split the input sequences into subsequences that can be processed by CNN.
# split our univariate time series data into input/output samples with four steps as input and one as output.
# Each sample can then be split into two sub-samples, each with two time steps. The CNN can interpret each sub-sequence
# of two time-steps and provide a time series of interpretations of the subsequences to the LSTM model to process as
# input.
# The input data can then be reshaped to have the required structure:
# [samples, subsequences, timesteps, features]
n_steps = 4
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape(X.shape[0], n_seq, n_steps, n_features)
# We want to reuse the same CNN model in a TimeDistributed wrapper that will appli the entire model once per input, in
# this case once per input subsequence
model = tf.keras.models.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
                                    input_shape=(None, n_steps, n_features)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=1)

x_input = np.array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
