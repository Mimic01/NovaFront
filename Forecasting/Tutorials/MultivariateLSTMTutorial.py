import numpy as np
import pandas as pd
import tensorflow as tf

# Multivariate time series data means data where there is more than one observation for each time step
# Two models we may require are Multiple input series and Multiple parallel series

# MULTIPLE INPUT SERIES

# A problem may have 2 or more parallel input time series and an output time series that is dependent on the input
# time series
# Input time series are parallel because each series has an observation at the same time steps

in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
# we can reshape these 3 arrays as a single dataset where each row is a time step and each column is a separate time
# series
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = np.hstack((in_seq1, in_seq2, out_seq))
print(dataset)


# This method will take the dataset which has rows for time steps and columns for parallel series and will return
# input/output samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_steps = 3

X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
for i in range(len(X)):
    print(X[i], y[i])

n_features = X.shape[2]

# the model expects as input the number of time steps and parallel series (features)
# when making a prediction the model expects 3 time steps for 2 input time series
# the shape of 1 sample with 3 time steps and 2 variables must be [1, 3, 2]
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=1)

x_input = np.array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=1)
print(yhat)


# MULTIPLE PARALLEL SERIES (multivariate forecasting)
# this is where there are multiple time series and a value must be predicted for each

# [[ 10  15  25]
#  [ 20  25  45]
#  [ 30  35  65]
#  [ 40  45  85]]

# input
# 10, 15, 25
# 20, 25, 45
# 30, 35, 65

# output
# 40, 45, 85

# this method will split multiple parallel time series with rows for time steps and one series per column into the
# required input/output shape
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


X, y = split_sequences(dataset, n_steps)
n_features = X.shape[2]
print(X.shape, y.shape)
# shape of X is 3-dim, number of samples(6), number of time-steps chosen per sample (3),
# number of parallel time series (features) (3)
# shape of y is 2-dim for number of samples (6) and number of time variables per sample to be predicted (3)

# data is ready to use in an LSTM model that expects 3-dim input and 2-dim output shapes for the X and y components of
# each sample

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
    tf.keras.layers.LSTM(100, activation='relu'),
    tf.keras.layers.Dense(n_features)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=400, verbose=1)

# The shape of the input for making a single prediction must be 1 sample, 3 time steps and 3 features: [1, 3, 3]
# 70, 75, 145
# 80, 85, 165
# 90, 95, 185

x_input = np.array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

