import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import tensorflow as tf
from matplotlib import pyplot as plt

# In multi-class classification, and instance of a record can belong to one and only one of the multiple output classes.
# Multi-label classification on the other hand can have multiple outputs at the same time.

toxic_comments = pd.read_csv(r"J:\Documents\Datahound\Datasets\wikipedia-toxiccomments\toxic-comments.csv")
print(toxic_comments.shape)
print(toxic_comments.head())

filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()

toxic_comments_labels = toxic_comments[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
print(toxic_comments_labels.head())

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
toxic_comments_labels.sum(axis=0).plot.bar()
plt.show()


# Approach #1 "Multi-lable text classification with single output layer" use a single dense layer with six outputs with
# a sigmoid activation function and binary cross entropy loss function. Each neuron in the output dense layer will
# represent one of the six output labels, sigmoid will have a value between 0 and 1 for each neuron.
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


X = []
sentences = list(toxic_comments["comment_text"])
for sen in sentences:
    X.append(preprocess_text(sen))
# No need to do one-hot encoding because output labels are already in that form
y = toxic_comments_labels.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=500)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 200

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
EMBEDDING_FILE = r"J:\Documents\Datahound\Datasets\quora-insincere-questions\glove.6B.50d.txt"

f = open(EMBEDDING_FILE, 'r', errors='ignore', encoding='utf8')
for line in f:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
f.close()

embedding_matrix = np.zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(maxlen,)),
    tf.keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.LSTM(128),
    # One dense layer with 6 neurons since we hace 6 labels
    tf.keras.layers.Dense(6, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


