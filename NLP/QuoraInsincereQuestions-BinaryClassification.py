import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
pd.set_option('display.max_columns', None)

train = pd.read_csv(r"J:\Documents\Datahound\Datasets\quora-insincere-questions\train.csv")
test = pd.read_csv(r"J:\Documents\Datahound\Datasets\quora-insincere-questions\test.csv")
train.question_text = train.question_text.astype("str")

print(train.info())
print(train.head())
# train.target.value_counts().plot(kind='bar', title='Count of target labels')
# plt.show()
# plt.close()
print(f"Maximum sequence length:  {train.question_text.apply(len).max()}")
print(f"Most frequent sequence length: {train.question_text.apply(len).mode()[0]}")
print(f"Mean sequence length: {train.question_text.apply(len).mean()}")

# train.question_text.apply(len).plot(kind='hist', bins=50, title="Histogram of question length")
# plt.show()
# plt.close()
# Set max sentence length based on right edge of question length histogram
maxlen = 250
# artibraty choice of top 25000 words
max_features = 25000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, oov_token="oov",
                                                  filters='"!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"', split=" ")
tokenizer.fit_on_texts(train.question_text)
train_df = tokenizer.texts_to_sequences(train.question_text)
train_df = tf.keras.preprocessing.sequence.pad_sequences(train_df, maxlen=maxlen, padding="post", truncating='post')

# Create a small 100k training sample and 5k validation sample
train_len = int(train_df.shape[0] * 0.1)
X_train = train_df[:train_len]
Y_train = train.target[:train_len]
X_val = train_df[train_len:train_len + 5000]
Y_val = train.target[train_len:train_len + 5000]

# Balance data on train set
rus = RandomUnderSampler(random_state=42)
X_balance, Y_balance = rus.fit_resample(X_train, Y_train.values)

EMBEDDING_FILE = r"J:\Documents\Datahound\Datasets\quora-insincere-questions\glove.6B.50d.txt"
embeddings_index = dict()
f = open(EMBEDDING_FILE, 'r', errors='ignore', encoding="utf8")
# load values into the embeddings_index dictionary filtering f or words not in corpus
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float')
    embeddings_index[word] = coefs

f.close()

# Get the mean and std of the embeddings weights
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

# Add the missing words to the embeddings and generate the random values
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print(f"Maximum sequence length: {maxlen}")
print(f"Number of words in the embedding: {max_features}")
print(f"Number of words in the vocabulary: {len(tokenizer.word_index)}")
print(f"Number of features per embedding: {embed_size}")


def sequence_model(maxlen, max_features, embed_size, embedding_matrix, metrics):
    tf.keras.backend.clear_session()
    if embedding_matrix is not None:
        embeddings = [embedding_matrix]
        output_dim = embedding_matrix.shape[1]
    else:
        embeddings = None

    model = tf.keras.models.Sequential([
        # maxlen = maximum sentence length
        tf.keras.Input(shape=maxlen),
        # max_features = artibraty choice of top 25000 words
        # embed_size = number of embeddings
        tf.keras.layers.Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(27, activation='tanh', return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(lr=0.00008, epsilon=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


METRICS = [
    # tf.keras.metrics.AUC(name='roc-auc'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name="recall")
]
model = sequence_model(maxlen, max_features, embed_size, embedding_matrix, METRICS)
hist = model.fit(X_balance, Y_balance, epochs=50, batch_size=64, validation_data=(X_val, Y_val))


def plot(history, *metrics):
    n_plots = len(metrics)
    fig, axs = plt.subplots(1, n_plots, figsize=(18, 5))
    hist = history.history

    for ax, metric in zip(axs, metrics):
        ax.plot(np.clip(hist[metric], 0, 1))
        ax.plot(np.clip(hist["val_" + metric], 0, 1))
        ax.legend([metric, "val_" + metric])
        ax.set_title(metric)
    plt.show()


plot(hist, 'loss', 'accuracy', 'precision', 'recall')

print(test.head())
X_test = tokenizer.texts_to_sequences(test.question_text)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')
preds = model.predict(X_test)
