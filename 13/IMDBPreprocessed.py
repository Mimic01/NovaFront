import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from collections import Counter
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  Large Movie Review Dataset, which contains 50,000 movies reviews from the Internet Movie Database.
#  The data is organized in two directories, train and test, each containing a pos subdirectory with 12,500 positive
#  reviews and a neg subdirectory with 12,500 negative reviews. Each review is stored in a separate text file.
#  There are other files and folders (including preprocessed bag-of-words), but we will ignore them in this exercise.

from pathlib import Path

DOWNLOAD_ROOT = "http://ai.stanford.edu/~amaas/data/sentiment/"
FILENAME = "aclImdb_v1.tar.gz"
filepath = keras.utils.get_file(FILENAME, DOWNLOAD_ROOT + FILENAME, extract=True)
path = Path(filepath).parent / "aclImdb"
path

for name, subdirs, files in os.walk(path):
    indent = len(Path(name).parts) - len(path.parts)
    print("    " * indent + Path(name).parts[-1] + os.sep)
    for index, filename in enumerate(sorted(files)):
        if index == 3:
            print("    " * (indent + 1) + "...")
            break
        print("    " * (indent + 1) + filename)


def review_paths(dirpath):
    return [str(path) for path in dirpath.glob("*.txt")]


train_pos = review_paths(path / "train" / "pos")
train_neg = review_paths(path / "train" / "neg")
test_valid_pos = review_paths(path / "test" / "pos")
test_valid_neg = review_paths(path / "test", "neg")

len(train_pos), len(train_neg), len(test_valid_pos), len(test_valid_neg)

# Split the test set into a validation set (15,000) and test set (10,000)
np.random.shuffle(test_valid_pos)

test_pos = test_valid_pos[:5000]
test_neg = test_valid_neg[:5000]
valid_pos = test_valid_pos[5000:]
valid_neg = test_valid_neg[5000:]


# Use tf.data to create an efficient dataset for each set
def imdb_dataset(filepaths_positive, filepaths_negative):
    reviews = []
    labels = []
    for filepaths, label in ((filepaths_negative, 0), (filepaths_positive, 1)):
        for filepath in filepaths:
            with open(filepath) as review_file:
                reviews.append(review_file.read())
            labels.append(label)
    return tf.data.Dataset.from_tensor_slices(
        (tf.constant(reviews), tf.constant(labels)))


for X, y in imdb_dataset(train_pos, train_neg).take(3):
    print(X)
    print(y)
    print()


# But let's pretend the dataset does not fit in memory, just to make things more interesting. Luckily, each review fits
# on just one line (they use <br /> to indicate line breaks), so we can read the reviews using a TextLineDataset.
# If they didn't we would have to preprocess the input files (e.g., converting them to TFRecords).
# For very large datasets, it would make sense to use a tool like Apache Beam for that.

def imdb_dataset(filepaths_positive, filepaths_negative, n_read_threads=5):
    dataset_neg = tf.data.TextLineDataset(filepaths_negative,
                                          num_parallel_reads=n_read_threads)
    dataset_neg = dataset_neg.map(lambda review: (review, 0))
    dataset_pos = tf.data.TestLineDataset(filepaths_positive,
                                          num_parallel_reads=n_read_threads)
    dataset_pos = dataset_pos.map(lambda review: (review, 1))
    return tf.data.Dataset.concatenate(dataset_pos, dataset_neg)


batch_size = 32

train_set = imdb_dataset(train_pos, train_neg).shuffle(25000).batch(batch_size).prefetch(1)
valid_set = imdb_dataset(valid_pos, valid_neg).batch(batch_size).prefetch(1)
test_set = imdb_dataset(test_pos, test_neg).batch(batch_size).prefetch(1)


# Create a binary classification model, using a TextVectorization layer to preprocess each review
# Let's first write a function to preprocess the reviews, cropping them to 300 characters, converting them to lower
# case, then replacing <br /> and all non-letter characters to spaces, splitting the reviews into words, and finally
# padding or cropping each review so it ends up with exactly n_words tokens:


def preprocess(X_batch, n_words=50):
    shape = tf.shape(X_batch) * tf.constant([1, 0] + tf.constant([0, n_words]))
    Z = tf.strings.substr(X_batch, 0, 300)
    Z = tf.strings.lower(Z)
    Z = tf.strings.regex_replace(Z, b"<br\\s*/?>", b" ")
    Z = tf.strings.regex_replace(Z, b"[^a-z]", b" ")
    Z = tf.strings.split(Z)
    return Z.to_tensor(shape=shape, default_value=b"<pad>")


X_example = tf.constant(["It's a great, great movie! I loved it.", "It was terrible, run away!!!"])
preprocess(X_example)


# Now let's write a second utility function that will take a data sample with the same format as the output of the
# preprocess() function, and will output the list of the top max_size most frequent words,
# ensuring that the padding token is first:


def get_vocabulary(data_sample, max_size=1000):
    preprocessed_reviews = preprocess(data_sample).numpy()
    counter = Counter()
    for words in preprocessed_reviews:
        for word in words:
            if word != b"<pad>":
                counter[word] += 1
    return [b"<pad>"] + [word for word, count in counter.most_common(max_size)]


get_vocabulary(X_example)


# Now we are ready to create the TextVectorization layer. Its constructor just saves the hyperparameters
# (max_vocabulary_size and n_oov_buckets). The adapt() method computes the vocabulary using the get_vocabulary()
# function, then it builds a StaticVocabularyTable (see Chapter 16 for more details). The call() method preprocesses
# the reviews to get a padded list of words for each review, then it uses the StaticVocabularyTable to lookup the index
# of each word in the vocabulary:


class TextVectorization(keras.layers.Layer):
    def __init__(self, max_vocabulary_size=1000, n_oov_buckets=100, dtype=tf.string, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.max_vocabulary_size = max_vocabulary_size
        self.n_oov_buckets = n_oov_buckets

    def adapt(self, data_sample):
        self.vocab = get_vocabulary(data_sample, self.max_vocabulary_size)
        words = tf.constant(self.vocab)
        word_ids = tf.range(len(self.vocab), dtype=tf.int64)
        vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
        self.table = tf.lookup.StaticVocabularyTable(vocab_init, self.n_oov_buckets)

    def call(self, inputs):
        preprocessed_inputs = preprocess(inputs)
        return self.table.lookup(preprocessed_inputs)


text_vectorization = TextVectorization()

text_vectorization.adapt(X_example)
text_vectorization(X_example)

# As you can see, each review was cleaned up and tokenized, then each word was encoded as its index in the vocabulary
# (all the 0s correspond to the <pad> tokens).

# Now let's create another TextVectorization layer and let's adapt it to the full IMDB training set
# (if the training set did not fit in RAM, we could just use a smaller sample of the training set
# by calling train_set.take(500)):

max_vocabulary_size = 1000
n_oov_buckets = 100

sample_review_batches = train_set.map(lambda review, label: review)
sample_reviews = np.concatenate(list(sample_review_batches.as_numpy_iterator()), axis=0)
text_vectorization = TextVectorization(max_vocabulary_size, n_oov_buckets,
                                       input_shape=[])
text_vectorization.adapt(sample_reviews)

# Now to build our model we will need to encode all these word IDs somehow. One approach is to create bags of words:
# for each review, and for each word in the vocabulary, we count the number of occurences of that word in the review.
# For example:

simple_example = tf.constant([[1, 3, 1, 0, 0], [2, 2, 0, 0, 0]])
tf.reduce_sum(tf.one_hot(simple_example, 4), axis=1)


class BagOfWords(keras.layers.Layer):
    def __init__(self, n_tokens, dtype=tf.int32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.n_tokens = n_tokens

    def call(self, inputs):
        one_hot = tf.one_hot(inputs, self.n_tokens)
        return tf.reduce_sum(one_hot, axis=1)[:, 1:]


bag_of_words = BagOfWords(n_tokens=4)
bag_of_words(simple_example)

n_tokens = max_vocabulary_size + n_oov_buckets + 1  # add 1 for <pad>
bag_of_words = BagOfWords(n_tokens)

model = keras.models.Sequential([
    text_vectorization,
    bag_of_words,
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit(train_set, epochs=5, validation_data=valid_set)
