import numpy as np
import pandas as pd
import tensorflow as tf


# Multiple input multi-step output
# Consider multivariate time-series:
# [[ 10  15  25]
#  [ 20  25  45]
#  [ 30  35  65]
#  [ 40  45  85]
#  [ 50  55 105]
#  [ 60  65 125]
#  [ 70  75 145]
#  [ 80  85 165]
#  [ 90  95 185]]
# we may use 3 prior time steps of each of the 2 input time series to predict 2 time steps of the output time series
# input
# 10, 15
# 20, 25
# 30, 35
# output
# 65
# 85
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# define input sequence
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = np.hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
# We can see that the shape of the input portion of the samples is three-dimensional, comprised of six samples,
# with three time steps, and two variables for the 2 input time series.
# The output portion of the samples is two-dimensional for the six samples and the two time steps for each
# sample to be predicted.
for i in range(len(X)):
    print(X[i], y[i])

n_features = X.shape[2]
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)),
    tf.keras.layers.LSTM(100, activation='relu'),
    tf.keras.layers.Dense(n_steps_out)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=1)

x_input = np.array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# Multiple Parallel input and multi-step output

# Consider the multivariate time series:
# [[ 10  15  25]
#  [ 20  25  45]
#  [ 30  35  65]
#  [ 40  45  85]
#  [ 50  55 105]
#  [ 60  65 125]
#  [ 80  85 165]
#  [ 70  75 145]
#  [ 90  95 185]]
# input
# 10, 15, 25
# 20, 25, 45
# 30, 35, 65
# output
# 40, 45, 85
# 50, 55, 105

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])

n_features = X.shape[2]
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)),
    tf.keras.layers.RepeatVector(n_steps_out),
    tf.keras.layers.LSTM(200, activation='relu', return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=300, verbose=1)

x_input = np.array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
