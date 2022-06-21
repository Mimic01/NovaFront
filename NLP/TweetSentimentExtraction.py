import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import plotly.graph_objects as go

# categorical_crossentropy works on one-hot encoded target, while sparse_categorical_crossentropy works on integer
# target

#  predict the sentiment based on the tweet
train = pd.read_csv(r"J:\Documents\Datahound\Datasets\tweet-sentiment-extraction\train.csv")
test = pd.read_csv(r"J:\Documents\Datahound\Datasets\tweet-sentiment-extraction\test.csv")
train = train.dropna()
print(train.head())
print(train.info())

# train.sentiment.value_counts().plot(kind='bar', title='Count of target labels')
# plt.show()
# plt.close()

print(f"Maximum sequence length: {train.text.apply(len).max()}")
print(f"Most frequent sentence length: {train.text.apply(len).mode()[0]}")
print(f"Mean sequence length: {train.text.apply(len).mean()}")

# train.text.apply(len).plot(kind='hist', bins=50, title='Histogram of tweet length')
# plt.show()
# plt.close()

maxlen = 150
max_features = 25000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, oov_token='oov',
                                                  filters='"!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"', split=" ")
tokenizer.fit_on_texts(train.text)
train_df = tokenizer.texts_to_sequences(train.text)
train_df = tf.keras.preprocessing.sequence.pad_sequences(train_df, maxlen=maxlen, padding="post", truncating='post')

X = train_df
y = train['sentiment']

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)

EMBEDDING_FILE = r"J:\Documents\Datahound\Datasets\quora-insincere-questions\glove.6B.50d.txt"
embedding_index = dict()
f = open(EMBEDDING_FILE, 'r', errors='ignore', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float')
    embedding_index[word] = coefs
f.close()

all_embs = np.stack(embedding_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print(f"Maximum sequence length: {maxlen}")
print(f"Number of words in the embedding: {max_features}")
print(f"Number of word in the vocabulary: {len(tokenizer.word_index)}")
print(f"Number of features per embedding: {embed_size}")


def sequence_model(maxlen, max_features, embed_size, embedding_matrix, metrics):
    if embedding_matrix is not None:
        embeddings = [embedding_matrix]
        output_dim = embedding_matrix.shape[1]
    else:
        embeddings = None

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=maxlen),
        tf.keras.layers.Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(800, activation='tanh', dropout=0.2, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(650, activation='tanh', dropout=0.2, return_sequences=True)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(256, activation='tanh', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
]
EPOCHS = 100
model = sequence_model(maxlen, max_features, embed_size, embedding_matrix, METRICS)
print(model.summary())

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, validation_data=(X_val, y_val))


def display_training_curves(training, validation, yaxis):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS+1), mode='lines+markers', y=training, marker=dict(color="dodgerblue"),
                   name='Train'))
    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS+1), mode='lines+markers', y=validation, marker=dict(color='darkorange'),
                   name="Val"))
    fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs", template="plotly_white")
    fig.show()


display_training_curves(
    history.history['categorical_accuracy'],
    history.history['val_categorical_accuracy'],
    'accuracy')

print(test.head())
X_test = tokenizer.texts_to_sequences(test.text)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')
preds = model.predict(X_test)
