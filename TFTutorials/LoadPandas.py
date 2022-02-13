import pandas as pd
import tensorflow as tf
import pprint

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
df = pd.read_csv(csv_file)
print(df.head())
print(df.dtypes)
target = df.pop('target')

# A DataFrame as an array
numeric_features_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_features_names]
numeric_features.head()

tf.convert_to_tensor(numeric_features)

# A DataFrame, interpreted as a single tensor, can be used directly as an argument to the Model.fit method.
normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(numeric_features)
print(normalizer(numeric_features.iloc[:3]))


def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


# When you pass the DataFrame as the x argument to Model.fit, Keras treats the DataFrame as it would a NumPy array:
model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

# With tf.data
# If you want to apply tf.data transformations to a DataFrame of a uniform dtype, the Dataset.from_tensor_slices method
# will create a dataset that iterates over the rows of the DataFrame. Each row is initially a vector of values.
# To train a model, you need (inputs, labels) pairs, so pass (features, labels) and Dataset.from_tensor_slices will
# return the needed pairs of slices:

numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))
for row in numeric_dataset.take(3):
    print(row)

numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)
model = get_basic_model()
model.fit(numeric_batches, epochs=15)

# A DataFrame as a dictionary (heterogeneus data)
# to make a dataset of dictionary-examples from a DataFrame,
# just cast it to a dict before slicing it with Dataset.from_tensor_slices
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))
for row in numeric_dict_ds.take(3):
    print(row)


# Typically, Keras models and layers expect a single input tensor, but these classes can accept and return nested
# structures of dictionaries, tuples and tensors. These structures are known as "nests"
# There are two equivalent ways you can write a keras model that accepts a dictionary as input

# 1. Model-subclass style
def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


# This model can accept either a dictionary of columns or a dataset of dictionary-elements for training:
# model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

# numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
# model.fit(numeric_dict_batches, epochs=5)
# model.predict(dict(numeric_features.iloc[:3]))

# 2. Keras functional style
inputs = {}
for name, column in numeric_features.items():
  inputs[name] = tf.keras.Input(
      shape=(1,), name=name, dtype=tf.float32)

print(inputs)


x = stack_dict(inputs, fun=tf.concat)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)

model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)


# FULL EXAMPLE

# Build preprocessing head
# If you have many features that need identical preprocessing it's more efficient to concatenate them together
# befofre applying the preprocessing.
# Binary features on the other hand do not generally need to be encoded or normalized.

binary_feature_names = ['sex', 'fbs', 'exang']
categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

# The next step is to build a preprocessing model that will apply apropriate preprocessing
# to each to each input and concatenate the results.

# This section uses the Keras Functional API to implement the preprocessing.
# You start by creating one tf.keras.Input for each column of the dataframe:
inputs = {}
for name, column in df.items():
    if type(column[0]) == str:
        dtype = tf.string
    elif (name in categorical_feature_names or name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

print(inputs)

# For each input you'll apply some transformations using Keras layers and TensorFlow ops. Each feature starts as a
# batch of scalars (shape=(batch,)). The output for each should be a batch of tf.float32 vectors (shape=(batch, n)).
# The last step will concatenate all those vectors together.

# Binary inputs
# Since the binary inputs don't need any preprocessing, just add the vector axis, cast them to float32
# and add them to the list of preprocessed inputs:
preprocessed = []

for name in binary_feature_names:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed.append(float_value)

print(preprocessed)

# Numeric inputs
# Run the numeric inputs through a normalization layer before using them
#  The code below collects the numeric features from the DataFrame, stacks them together and passes those to the
#  Normalization.adapt method.
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))
# The code below stacks the numeric features and runs them through the normalization layer
numeric_inputs = {}
for name in numeric_features_names:
    numeric_inputs[name]=inputs[name]
numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)
preprocessed.append(numeric_normalized)

print(preprocessed)

# Categorical features
# To use categorical features you'll first need to encode them into either binary vectors or embeddings
# Let's convert the inputs directly to one-hot vectors using the output_mode='one_hot' option, supported byy both the
# tf.keras.layers.StringLookup and tf.keras.layers.IntegerLookup layers.
vocab = ['a', 'b', 'c']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup(['c', 'a', 'a', 'b', 'zzz']))

vocab = [1,4,7,99]
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup([-1,4,1]))

# To determine the vocabulary for each input, create a layer to convert that vocabulary to a one-hot vector:
for name in categorical_feature_names:
    vocab = sorted(set(df[name]))
    print(f'name: {name}')
    print(f'vocab: {vocab}\n')

    if type(vocab[0]) is str:
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
    else:
        lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
    x = inputs[name][:, tf.newaxis]
    x = lookup(x)
    preprocessed.append(x)

# Assemble preprocessing head
# At this point preprocessed is just a Python list of all the preprocessing results,
# each result has a shape of (batch_size, depth):
print(preprocessed)

# Concatenate all the preprocessed features along the depth axis, so each dictionary-example is converted into a
# single vector. The vector contains categorical features, numeric features, and categorical one-hot features:
preprocessed_result = tf.concat(preprocessed, axis=-1)
print(preprocessed_result)
# Now create a model out of that calculation so it can be reused:
preprocessor = tf.keras.Model(inputs, preprocessed_result)

tf.keras.utils.plot_model(preprocessor, rankdir="LR", show_shapes=True)

# To test the preprocessor, use the DataFrame.iloc accessor to slice the first example from the DataFrame.
# Then convert it to a dictionary and pass the dictionary to the preprocessor.
# The result is a single vector containing the binary features, normalized numeric features and the one-hot categorical
# features, in that order:
preprocessor(dict(df.iloc[:1]))

# Create and train a model
body = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Now put the two pieces together using the Keras functional API
print(inputs)

x = preprocessor(inputs)
print(x)

result = body(x)
print(result)

model = tf.keras.Model(inputs, result)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)

# Using tf.data works as well:
ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
ds = ds.batch(BATCH_SIZE)

for x, y in ds.take(1):
    pprint.pprint(x)
    print()
    print(y)

history = model.fit(ds, epochs=5)