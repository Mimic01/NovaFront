import collections
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = pathlib.Path(text_dir).parent
print(list(parent_dir.iterdir()))


# Here, you will use tf.data.TextLineDataset, which is designed to create a tf.data.Dataset from a text file where
# each example is a line of text from the original file. TextLineDataset is useful for text data that is primarily
# line-based (for example, poetry or error logs).
def labeler(example, index):
    return example, tf.cast(index, tf.int64)


labeled_data_sets = []
for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(str(parent_dir / file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

# Next, you'll combine these labeled datasets into a single dataset using Dataset.concatenate,
# and shuffle it with Dataset.shuffle:
BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)


# you will now use the TensorFlow Text APIs to standardize and tokenize the data, build a vocabulary and use
# tf.lookup.StaticVocabularyTable to map tokens to integers to feed to the model.
tokenizer = tf_text.UnicodeScriptTokenizer()


def tokenize(text, unused_label):
    lower_case = tf_text.case_fold_utf8(text)
    return tokenizer.tokenize(lower_case)


tokenized_ds = all_labeled_data.map(tokenize)

for text_batch in tokenized_ds.take(5):
    print("Tokens: ", text_batch.numpy())

# Next, you will build a vocabulary by sorting tokens by frequency and keeping the top VOCAB_SIZE tokens:
AUTOTUNE = tf.data.AUTOTUNE


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


tokenized_ds = configure_dataset(tokenized_ds)
VOCAB_SIZE = 10000

vocab_dict = collections.defaultdict(lambda: 0)
for toks in tokenized_ds.as_numpy_iterator():
    for tok in toks:
        vocab_dict[tok] += 1
vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for token, count in vocab]
vocab = vocab[:VOCAB_SIZE]
vocab_size = len(vocab)
print("Vocab size: ", vocab_size)
print("First five vocab entries: ", vocab[:5])

# To convert the tokens into integers, use the vocab set to create a tf.lookup.StaticVocabularyTable.
# You will map tokens to integers in the range [2, vocab_size + 2]. As with the TextVectorization layer, 0 is
# reserved to denote padding and 1 is reserved to denote an out-of-vocabulary (OOV) token.
keys = vocab
values = range(2, len(vocab) + 2)
init = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64)
num_oov_buckets = 1
vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)


# Finally, define a function to standardize, tokenize and vectorize the dataset using the tokenizer and lookup table:
def preprocess_text(text, label):
    standarized = tf_text.case_fold_utf8(text)
    tokenized = tokenizer.tokenize(standarized)
    vectorized = vocab_table.lookup(tokenized)
    return vectorized, label


all_encoded_data = all_labeled_data.map(preprocess_text)

# padding and splitting
train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
validation_data = all_encoded_data.take(VALIDATION_SIZE)

train_data = train_data.padded_batch(BATCH_SIZE)
validation_data = validation_data.padded_batch(BATCH_SIZE)

# Now, validation_data and train_data are not collections of (example, label) pairs, but collections of batches.
# Each batch is a pair of (many examples, many labels) represented as arrays.
sample_text, sample_labels = next(iter(validation_data))
print("Text batch shape: ", sample_text.shape)
print("Label batch shape: ", sample_labels.shape)
print("First text example: ", sample_text[0])
print("First label example: ", sample_labels[0])

# Since you use 0 for padding and 1 for out-of-vocabulary (OOV) tokens, the vocabulary size has increased by two:
vocab_size += 2

train_data = configure_dataset(train_data)
validation_data = configure_dataset(validation_data)


def create_model(vocab_size, num_labels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
        tf.keras.layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(num_labels)
    ])
    return model


model = create_model(vocab_size, num_labels=3)
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_data, validation_data=validation_data, epochs=3)

loss, accuracy = model.evaluate(validation_data)
print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))
