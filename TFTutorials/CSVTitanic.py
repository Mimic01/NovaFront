import pandas as pd
import numpy as np
import pathlib
import re
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import itertools
np.set_printoptions(precision=3, suppress=True)

# IN MEMORY

# Titanic Dataset (mixed data types)

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
print(titanic.head())

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

#  Build a set of symbolic keras.Input objects, matching the names and data-types of the CSV columns.
inputs = {}
for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

# concatenate the numeric inputs together, and run them through a normalization layer:
numeric_inputs = {name: input for name, input in inputs.items()
                  if input.dtype == tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

# Collect all the symbolic preprocessing results, to concatenate them later.
preprocessed_inputs = [all_numeric_inputs]

# For the string inputs use the tf.keras.layers.StringLookup function to map from strings to integer indices in a
# vocabulary. Next, use tf.keras.layers.CategoryEncoding to convert the indexes into float32 data
# appropriate for the model.

# The default settings for the tf.keras.layers.CategoryEncoding layer create a one-hot vector for each input
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

# With the collection of inputs and processed_inputs, you can concatenate all the preprocessed inputs together,
# and build a model that handles the preprocessing:
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir="LR", dpi=72, show_shapes=True)

#  Keras models don't automatically convert Pandas DataFrames because it's not clear if it should be converted to
#  one tensor or to a dictionary of tensors. So convert it to a dictionary of tensors:
titanic_features_dict = {name: np.array(value) for name, value in titanic_features.items()}
features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)


# Now build the model on top of this:
def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.optimizers.Adam())
    return model


titanic_model = titanic_model(titanic_preprocessing, inputs)
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

# Since the preprocessing is part of the model, you can save the model and reload it somewhere else
# and get identical results:
titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')

features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before - after) < 1e-3
print(before)
print(after)


# tf.data
# Manually slice up the titanic dictionary just created. For each index, it take that index for each feature:
def slices(features):
    for i in itertools.count():
        example = {name: values[i] for name, values in features.items()}
        yield example


for example in slices(titanic_features_dict):
    for name, value in example.items():
        print(f"{name:19s}: {value}")
    break

# The most basic tf.data.Dataset in memory data loader is the Dataset.from_tensor_slices constructor.
# This returns a tf.data.Dataset that implements a generalized version of the above slices function, in TensorFlow.
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)

for example in features_ds:
    for name, value in example.items():
        print(f"{name:19s}: {value}")
    break

# The from_tensor_slices function can handle any structure of nested dictionaries or tuples.
# The following code makes a dataset of (features_dict, labels) pairs:
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)
titanic_model.fit(titanic_batches, epochs=5)

# FROM A SINGLE FILE
titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_csv_ds = tf.data.experimental.make_csv_dataset(titanic_file_path, batch_size=5, label_name='survived',
                                                       num_epochs=1, ignore_errors=True)
for batch, label in titanic_csv_ds.take(1):
    for key, value in batch.items():
        print(f"{key:20s}: {value}")
    print()
    print(f"{'label':20s}: {label}")

# You can also decompress data on the fly
# e.g. gzipped CSV of metro intersate traffic dataset
traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz',
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir='.', cache_subdir='traffic')

# Set the compression_type argument to read directly from the compressed file:
traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size=256,
    label_name='traffic_volume',
    num_epochs=1,
    compression_type="GZIP"
)
for batch, label in traffic_volume_csv_gz_ds.take(1):
    for key, value in batch.items():
        print(f"{key:20s}: {value[:5]}")
    print()
    print(f"{'label' :20s}: {label[:5]}")

# MULTIPLE FILES
fonts_zip = tf.keras.utils.get_file(
    'fonts.zip', "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True)

font_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))
print(font_csvs[:10])
print(len(font_csvs))

fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern="fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)

# These csv files have the images flattened out into a single row. The column names are formatted r{row}c{column}.
# Here's the first batch:
for features in fonts_ds.take(1):
    for i, (name, value) in enumerate(features.items()):
        if i > 15:
            break
        print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")


# Packing fields
# You probably don't want to work with each pixel in separate columns like this.
# Before trying to use this dataset be sure to pack the pixels into an image-tensor.
# Here is code that parses the column names to build images for each example:
def make_images(features):
    image = [None] * 400
    new_feats = {}

    for name, value in features.items():
        match = re.match('r(\d+)c(\d+)', name)
        if match:
            image[int(match.group(1)) * 20 + int(match.group(2))] = value
        else:
            new_feats[name] = value
    image = tf.stack(image, axis=0)
    image = tf.reshape(image, [20, 20, -1])
    new_feats['image'] = image

    return new_feats


# Apply this to every batch
fonts_image_ds = fonts_ds.map(make_images)
for features in fonts_image_ds.take(1):
    break

plt.figure(figsize=(6, 6), dpi=120)
for n in range(9):
    plt.subplot(3, 3, n + 1)
    plt.imshow(features['image'][..., n])
    plt.title(chr(features['m_label'][n]))
    plt.axis('off')

# Lower level functions
# tf.io.decode_csv - a function for parsing lines of text into a list of CSV column tensors.
text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]
all_strings = [str()] * 10
print(all_strings)

features = tf.io.decode_csv(lines, record_defaults=all_strings)
for f in features:
    print(f"type: {f.dtype.name}, shape: {f.shape}")

# To parse them with their actual types, create a list of record_defaults of the corresponding types:
print(lines[0])
titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
print(titanic_types)

features = tf.io.decode_csv(lines, record_defaults=titanic_types)
for f in features:
    print(f"type: {f.dtype.name}, shape: {f.shape}")

# tf.data.experimental.CsvDataset - a lower level csv dataset constructor
simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)
for example in simple_titanic.take(1):
    print([e.numpy() for e in example])

# Multiple files
# To parse the fonts dataset using experimental.CsvDataset
font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)
#  get the total number of features by counting the commas:
num_font_features = font_line.count(',') + 1
font_column_types = [str(), str()] + [float()] * (num_font_features - 2)
print(font_csvs[0])

simple_font_ds = tf.data.experimental.CsvDataset(font_csvs, record_defaults=font_column_types, header=True)
for row in simple_font_ds.take(10):
    print(row[0].numpy())

font_files = tf.data.Dataset.list_files("fonts/*.csv")

# This shuffles the file names each epoch:
print('Epoch 1:')
for f in list(font_files)[:5]:
    print("    ", f.numpy())
print('    ...')
print()

print('Epoch 2:')
for f in list(font_files)[:5]:
    print("    ", f.numpy())
print('    ...')


# The interleave method takes a map_func that creates a child-Dataset for each element of the parent-Dataset.
# Here, you want to create a CsvDataset from each element of the dataset of files:

def make_font_csv_ds(path):
    return tf.data.experimental.CsvDataset(path, record_defaults=font_column_types, header=True)


font_rows = font_files.interleave(make_font_csv_ds, cycle_length=3)
fonts_dict = {'font_name': [], 'character': []}

for row in font_rows.take(10):
    fonts_dict['font_name'].append(row[0].numpy().decode())
    fonts_dict['character'].append(chr(row[2].numpy()))
pd.DataFrame(fonts_dict)

# io.decode_csv is more efficient when run on a batch of strings.
#
# It is possible to take advantage of this fact, when using large batch sizes, to improve CSV loading performance
# (but try caching first).
BATCH_SIZE = 2048
fonts_ds = tf.data.experimental.make_csv_dataset(file_pattern='fonts/*.csv', batch_size=BATCH_SIZE, num_epochs=1,
                                                 num_parallel_reads=100)
for i, batch in enumerate(fonts_ds.take(20)):
    print('.', end='')
print()

# Passing batches of text lines todecode_csv runs faster, in about 5s:
fonts_files = tf.data.Dataset.list_files("fonts/*.csv")
fonts_lines = fonts_files.interleave(lambda fname: tf.data.TextLineDataset(fname).skip(1),
                                     cycle_length=100).batch(BATCH_SIZE)
fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))

for i, batch in enumerate(fonts_fast.take(20)):
    print('.', end='')
print()
