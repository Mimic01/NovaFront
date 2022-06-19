import pandas as pd

# Given a DataFrame, the shift() function can be used to create copies of columns that are pushed forward
# (rows of NaN values added to the front) or pulled back (rows of NaN values added to the end).
# This is the behavior required to create columns of lag observations as well as columns of forecast observations for
# a time series dataset in a supervised learning format.

# The function is defined with default parameters so that if you call it with just your data, it will construct a
# DataFrame with t-1 as X and t as y.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n ... t+n)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with nan values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# One step univariate forecasting
# use lagged observations (e.g. t-1) as input variables to forecast the current time step (t).
values = [x for x in range(10)]
data = series_to_supervised(values)
print(data)

# We can repeat this example with an arbitrary number length input sequence, such as 3.
# This can be done by specifying the length of the input sequence as an argument; for example
data = series_to_supervised(values, 3)
print(data)

# Multi step forecasting
# A different type of forecasting problem is using past observations to forecast a sequence of future observations.
# We can frame a time series for sequence forecasting by specifying another argument. For example, we could frame a
# forecast problem with an input sequence of 2 past observations to forecast 2 future observations as follows:
data = series_to_supervised(values, 2, 2)
# Running the example shows the differentiation of input (t-n) and output (t+n) variables with the current
# observation (t) considered an output.
print(data)

# Multivariate forecasting
# This is where we may have observations of multiple different measures and an interest in
# forecasting one or more of them.
raw = pd.DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values)
print(data)

# Again, depending on the specifics of the problem, the division of columns into X and Y components
# can be chosen arbitrarily, such as if the current observation of var1 was also provided as input and only var2
# was to be predicted


