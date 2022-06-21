import pandas as pd
import matplotlib.pyplot as plt

series = pd.read_csv(r'J:\Documents\Datahound\Datasets\female-birth\daily-total-female-births.csv', header=0,
                      index_col=0)
print(series.head())

# Smoothing is useful as a data preparation technique as it can reduce the random variation in the observations and
# better expose the structure of the underlying causal processes.

# The rolling() function on the Series Pandas object will automatically group observations into a window.
# You can specify the window size, and by default a trailing window is created. Once the window is created,
# we can take the mean value, and this is our transformed dataset.
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))

# see the lag in the transformed dataset
plt.plot(series)
plt.plot(rolling_mean, color='red')
plt.show()

# Below is an example of including the moving average of the previous 3 values as a new feature, as wellas a lag-1
# input feature for the Daily Female Births dataset.

df = pd.DataFrame(series.values)
width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = pd.concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))