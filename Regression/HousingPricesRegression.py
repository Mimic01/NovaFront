import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

dataPath = "J:/Documents/Datahound/Datasets/housing-prices/HousingPrices.csv"
df = pd.read_csv(dataPath)
print(df.sample(5))

to_drop = ['date', 'street', 'statezip', 'country']
df = df.drop(to_drop, axis=1)
print(df.head())

# how old is the house
df['house_age'] = [2022 - yr_built for yr_built in df['yr_built']]

# Was the house renovated and was the renovation recent
df['was_renovated'] = [1 if yr_renovated != 0 else 0 for yr_renovated in df['yr_renovated']]
df['was_renovated_10_yrs'] = [1 if (2022 - yr_renovated) <= 10 else 0 for yr_renovated in df['yr_renovated']]
df['was_renovated_30_yrs'] = [1 if 10 < (2022 - yr_renovated) <= 30 else 0 for yr_renovated in df['yr_renovated']]

# drop original columns
df = df.drop(['yr_built', 'yr_renovated'], axis=1)
print(df.head())


def remap_location(data: pd.DataFrame, location: str, threshold: int = 50) -> str:
    if len(data[data['city'] == location]) < threshold:
        return 'Rare'
    return location


df['city'] = df['city'].apply(lambda x: remap_location(data=df, location=x))
print(df.sample(10))

# Check target variable distribution
rcParams['figure.figsize'] = (16, 6)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.hist(df['price'], bins=100)

# Handle outliers with Z-scores
df['price_z'] = np.abs(stats.zscore(df['price']))
# Filter out outliers
df = df[df['price_z'] <= 3]
# Remove hourses listed for $0
df = df[df['price'] != 0]
# Drop the column
df = df.drop('price_z', axis=1)
# Draw histogram
plt.hist(df['price'], bins=100)

# You can use make_column_transformer() function from scikit-learn to apply scaling and encoding in one go
transformer = make_column_transformer(
    (MinMaxScaler(),
     ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'house_age']),
    (OneHotEncoder(handle_unknown='ignore'),
     ['bedrooms', 'bathrooms', 'floors', 'view', 'condition'])
)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
transformer.fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

X_train = X_train.toarray()
X_test = X_test.toarray()


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = Sequential([
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1)
])
model.compile(loss=rmse, optimizer=Adam(), metrics=[rmse])
model.fit(X_train, y_train, epochs=100)

predictions = model.predict(X_test)
print(predictions[:5])
predictions = np.ravel(predictions)
print(predictions[:5])

rmse(y_test, predictions).numpy()

