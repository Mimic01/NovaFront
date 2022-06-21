import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

sns.set_style('dark')
from datetime import datetime, timedelta
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Multiple input, multiple output configuration such that each input and output is the univariate sequence of a given
# hour of the day
# The model makes a prediction each 24 hours at midnight and forecasts the next 24 hours of demand.
data = pd.read_csv(r"J:\Documents\Datahound\Datasets\energy-forecast\energy_dataset.csv", index_col=[0],
                   parse_dates=True)
print(data.head())
dates = pd.date_range(start='2014-12-31T2300', end='2018-12-31T2200', freq='H')
data.index = pd.DatetimeIndex(dates).tz_localize('UTC').tz_convert('Europe/Madrid')
df = data[['total load actual', 'total load forecast']]
null_vals = df.isnull().sum()
print('Null values in the target column {}'.format(null_vals))
print(df.head())
print(df.index.min())
print(df.index.max())

df = df.interpolate(method='linear', axis=0)
print(df.isnull().sum())


# partial/autocorrelation analysis
def transform_to_hour_cols(series):
    df = pd.DataFrame()
    start = series.index.min()
    end = series.index.max()

    df['year'] = series.index.year
    df['month'] = series.index.month
    df['day'] = series.index.day
    df['hours'] = series.index.hour
    df['loads'] = series.values

    df = df.set_index(['year', 'month', 'day', 'hours'], append=True).unstack()
    df = df.groupby(['year', 'month', 'day']).sum()
    date_list = pd.date_range(start=start, end=end, freq='D').strftime('%Y-%m-%d')
    df.index = pd.DatetimeIndex(date_list, name='date')

    return df


# Missing values were accounted for before the previous step. However the above transformation will introduce
# new missing values because of daylight savings time.
day_energy = transform_to_hour_cols(df['total load actual'])
print(day_energy.head())

day_energy.columns = ["h" + str(x) for x in range(0, 24)]
# get dates for daylight savings times
idx = day_energy.loc[day_energy['h2'] == 0, 'h2'].index
# set values zero values to NaN
day_energy.loc[day_energy['h2'] == 0, 'h2'] = np.NaN
print(day_energy.loc[idx, 'h2'])

day_energy = day_energy.interpolate(method='linear', axis=0)
print(day_energy.loc[idx, 'h2'])

# Autocorrelation and partial-autocorrelation analysis
# Energy demand as hourly-sequental view. I.e. Day 1 h1, h2...h23, Day 2, h1, h2...
# Energy demand as hour-by-hour transformation I.e. Day1, h1, Day 2, h1 ... Day 365 h1 for each hour in the day.

# isolate the original series of demand data
energy_demand_univar = df['total load actual']
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
# plot the last 30 and 90 days
lags = [30 * 24, 90 * 24]

# for ax, lag in zip(axs.flatten(), lags):
#     plot_acf(energy_demand_univar, ax=ax, lags=lag)
# plt.plot()
# plt.show()

# Autocorrelation hour
# plots = len(day_energy.columns)
# fig, axs = plt.subplots(int(plots / 2), 2, figsize=(15, 2 * plots))
# for hour, ax in zip(day_energy.columns, axs.flatten()):
#     plot_acf(day_energy.loc[:, hour], ax=ax, lags=60)
#     ax.set_title('Autocorrelation hour ' + str(hour))
# plt.plot()
# plt.show()

# Partial autocorrelation hour
# plots = len(day_energy.columns)
# fig, axs = plt.subplots(int(plots / 2), 2, figsize=(15, 2 * plots))
# for hour, ax in zip(day_energy.columns, axs.flatten()):
#     plot_pacf(day_energy.loc[:, hour], ax=ax, lags=60)
#     ax.set_title('Partial Autocorrelation Hour ' + str(hour))
# plt.plot()
# plt.show()


# X is composed of rows of multiple days (referenced as lags), and columns of the hours of the day
# y is then the 24 hours of observed energy demand for the target day.

def normalize_df(data):
    scaler = MinMaxScaler().fit(data.values)
    data_normd = scaler.transform(data.values)
    data = pd.DataFrame(data_normd, index=data.index, columns=data.columns)
    return data, scaler


day_energy_normed, scaler = normalize_df(day_energy)


def split_sequences(sequences, n_steps, extra_lag=False, long_lag_step=7, max_step=30, idx=0, multivar=False):
    if not extra_lag:
        max_step = n_steps
        n_steps += 1

    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + max_step
        slices = [x for x in range(end_ix - 1, end_ix - n_steps, -1)] + \
                 [y for y in range(end_ix - n_steps, i, -long_lag_step)]
        slices = list(reversed(slices))
        if end_ix > len(sequences) - 1:
            break
        seq_x = sequences[slices, :]
        seq_y = sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)

    if multivar:
        # unstack the 3rd dimension and select the first element (energy load)
        y = y[:, idx]
    return X, y


# Following the comments in the (partial)autocorrelation analysis we will use a daily lookback of 21 days to account
# for the hours 0-1 and 21-23. After 21 days, we will use multiples of 7 up to a max of 60 days.
n_steps = 21
X, y = split_sequences(day_energy_normed.values, n_steps, extra_lag=True, long_lag_step=7, max_step=60, idx=0,
                       multivar=False)
print(X.shape, y.shape)
print(X[:5], y[:5])


# takes in parallel inputs and outputs an equal number of parallel outputs
def lstm_parallel_out(n_lags, n_hours, cells=50, learning_rate=5e-3):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(cells, activation='relu', return_sequences=True, input_shape=(n_lags, n_hours)),
        tf.keras.layers.LSTM(int(cells/2), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_hours)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model


# We are training on data from 2015 to 2018 inclusive. This is a relatively small amount of energy data.
# Therefore we use time series split to cross validate within the 4 year period.
def crossval_testbench(X, y, n_crossvals, epochs=5, verbose=0):
    n_hours = X.shape[-1]
    n_features = X.shape[1]
    tscv = TimeSeriesSplit(n_splits=n_crossvals)

    predictions = []
    actuals = []

    # run the LSTM model on each of the time series splits
    for train, test in tscv.split(X, y):
        lstm_base = lstm_parallel_out(n_features, n_hours, learning_rate=5e-3)
        lstm_base.fit(X[train], y[train], epochs=epochs, verbose=verbose, shuffle=False)
        predict = lstm_base.predict(X[test], verbose=verbose)

        # inverse transform the predictions and actual values
        prediction = scaler.inverse_transform(predict)
        actual = scaler.inverse_transform(y[test].copy())

        predictions.append(prediction)
        actuals.append(actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return predictions, actuals


preds, actuals = crossval_testbench(X, y, 2, epochs=150, verbose=1)
print(preds.shape, actuals.shape)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# MAPE for a specific hour
error_h0 = mean_absolute_percentage_error(actuals[0, :5, 0], preds[0, :5, 0])
print(f'MAPE for Zero Hour: {round(error_h0, 2)}')

crossvals = actuals.shape[0]
hours = actuals.shape[2]

errors_crossvals = list()
for crossval in range(crossvals):
    errors_hourly = [mean_absolute_percentage_error(actuals[crossval, :, hour], preds[crossval, :, hour])
                     for hour in range(hours)]
    errors_crossvals.append(errors_hourly)
    errors = pd.DataFrame(errors_crossvals)
    errors['mean'] = errors.mean(axis=1)
    errors.index.name='crossval_set'
    errors.columns.name='hours'
    print(errors)

plt.figure(figsize=(8, 6))
plt.plot(errors.drop(columns='mean').T)
plt.title('MAPE per hourly prediction')
plt.legend(errors.index, title='crossval set')
plt.xlabel('Hour of the day')
plt.ylabel('MAPE')
plt.show()


# Predicting on an unseen test set
# The test bench allowed to modify and iterate on the model. To get an evaluation on model performance with unknown data
# we create a typical train test split.
# To do this we will also need to run the train and test sets through the preprocessing pipeline again.

def train_test_split(df, split_date):
    train_date = pd.Timestamp(split_date).strftime('%Y-%m-%d')
    test_date = (pd.Timestamp(split_date) + timedelta(1)).strftime('%Y-%m-%d')
    df_train = df[:train_date]
    df_test = df[test_date:]
    return df_train, df_test


train, test = train_test_split(day_energy, '2017-12-31')

print(f'Training start date {train.index.min()} end date {train.index.max()}')
print(f'Training start date {test.index.min()} end date {test.index.max()}')

train_norm, scalar = normalize_df(train)
test_norm = scalar.transform(test)

n_steps = 21
X_train, y_train = split_sequences(train_norm.values, n_steps, extra_lag=True, long_lag_step=7, max_step=60, idx=0,
                                   multivar=False)
print(f'Training Set X {X_train.shape} and Y {y_train.shape}')

# To construct the testing set samples we must add the last 60 (corresponding to max_step) values to the train.
# These will be used to make the first prediction in the test set.
test_set = np.vstack([train_norm.values[-60:], test_norm])
print(f'Dimensions of the test set with training data needed for predictions: {test_set.shape}')
X_test, y_test = split_sequences(test_set, n_steps, extra_lag=True, long_lag_step=7, max_step=60, idx=0, multivar=False)
print(f'Testing Set X {X_test.shape} and y {y_test.shape}')

# Use the entire training set for the model to learn
n_features = X_train.shape[1]
n_hours = X_train.shape[2]
lstm_eval = lstm_parallel_out(n_features, n_hours, learning_rate=5e-3)
lstm_eval.fit(X_train, y_train, epochs=350, verbose=1, shuffle=False)
train_predictions = lstm_eval.predict(X_train, verbose=1)
test_predictions = lstm_eval.predict(X_test, verbose=1)

# Rescale predictions and evaluate
train_preds = scalar.inverse_transform(train_predictions)
test_preds = scalar.inverse_transform(test_predictions)
y_train = scalar.inverse_transform(y_train)
y_test = scalar.inverse_transform(y_test)

train_error = pd.Datagrame([mean_absolute_percentage_error(y_train[:, hour], train_preds[:, hour])
                            for hour in range(hours)], columns=['train'])
test_error = pd.DataFrame([mean_absolute_percentage_error(y_test[:, hour], test_preds[:, hour])
                           for hour in range(hours)], columns=['test'])
errors = pd.concat([train_error, test_error], axis=1)
errors.index.name = 'hour'
errors.plot()
plt.show()

test_df = pd.DataFrame(test_preds).stack()
y_test_df = pd.DataFrame(y_test).stack()
preds_df = pd.concat([y_test_df, test_df], axis=1)
preds_df.columns = ['actual', 'predicted']
preds_df.index = pd.DatetimeIndex(pd.date_range(start='2018-01-01T0000', end='2018-12-31T2300', freq='H'))
fig = plt.figure(figsize=(10,10))
for week in range(52):
    fig.add_subplot()
    preds_df.iloc[week*7*24:(week+1)*7*24].plot()
    plt.title(f'Consumption profile for 2018 week: {week+1}')