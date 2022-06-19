from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

yelp_reviews = pd.read_csv(r"J:\Documents\Datahound\Datasets\yelp-reviews\yelp_review.csv")
EMBEDDING_FILE = r"J:\Documents\Datahound\Datasets\quora-insincere-questions\glove.6B.50d.txt"

yelp_reviews = yelp_reviews[0:50000]
bins = [0, 1, 3, 5]
review_names = ['bad', 'average', 'good']
yelp_reviews['reviews_score'] = pd.cut(yelp_reviews['stars'], bins=bins, labels=review_names)
yelp_reviews.isnull().values.any()
print(yelp_reviews.shape)
print(yelp_reviews.head())

sns.countplot(x='reviews_score', data=yelp_reviews)


# Model #1 Text input only model
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


X = []
sentences = list(yelp_reviews["text"])
for sen in sentences:
    X.append(preprocess_text(sen))
y = yelp_reviews['reviews_score']

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
# X_text = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
f = open(EMBEDDING_FILE, 'r', errors='ignore', encoding='utf8')
for line in f:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[:1], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
f.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# Using Keras functional API
# Model with one input layer, ons LSTM, and output layer with 3 outputs
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(3, activation='softmax')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

# score = model.evaluate(X_test, y_test, verbose=1)
# print("Test Score:", score[0])
# print("Test Accuracy:". score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Model #2 with meta information only
sns.barplot(x='reviews_score', y='useful', data=yelp_reviews)
sns.barplot(x='reviews_score', y='funny', data=yelp_reviews)
sns.barplot(x='reviews_score', y='cool', data=yelp_reviews)

yelp_reviews_meta = yelp_reviews[['useful', 'funny', 'cool']]
X = yelp_reviews_meta.values
y = yelp_reviews['reviews_score']

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

input2 = Input(shape=(3,))
dense_layer_1 = Dense(10, activation='relu')(input2)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(3, activation='softmax')(dense_layer_2)
model = Model(inputs=input2, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_split=0.2)

# score = model.evaluate(X_test, y_test, verbose=1)
# print("Test score: ", score[0])
# print("Test Accuracy: ", score[1])

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

# Model #3 Multiple Inputs (textual data and metainfo)
X = yelp_reviews.drop('reviews_score', axis=1)
y = yelp_reviews['reviews_score']

label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


def preprocess_text(sen):

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


X1_train = []
sentences = list(X_train["text"])
for sen in sentences:
    X1_train.append(preprocess_text(sen))

# X1_test = []
# sentences = list(X_test["text"])
# for sen in sentences:
#     X1_test.append(preprocess_text(sen))


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X1_train)

X1_train = tokenizer.texts_to_sequences(X1_train)
X1_test = tokenizer.texts_to_sequences(X1_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 200

X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
f = open(EMBEDDING_FILE, errors='ignore', encoding='utf8')
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


X2_train = X_train[['useful', 'funny', 'cool']].values
# X2_test = X_test[['useful', 'funny', 'fool']].values

# Submodel #1
input_1 = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input_1)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
# Submodel #2
input_2 = Input(shape=(3,))
dense_layer_1 = Dense(10, activation='relu')(input_2)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)

# Concatenate output from the first submodel with the output from the second submodel
# Concatenated model
concat_layer = Concatenate()([LSTM_Layer_1, dense_layer_2])
dense_layer_3 = Dense(10, activation='relu')(concat_layer)
output = Dense(3, activation='softmax')(dense_layer_3)
model = Model(inputs=[input_1, input_2], outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
tf.keras.utils.plot_model(model, to_file='concatenated_model.png', show_shapes=True, show_layer_names=True)

history = model.fit(x=[X1_train, X2_train], y=y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

score = model.evaluate(x=[X1_test, X2_test], y=y_test, verbose=1)
print("Test Score: ", score[0])
print("Test Accuracy: ", score[1])

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