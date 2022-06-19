import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
import re
import nltk
import nltk

nltk.download('gutenberg')
nltk.download('punkt')
macbeth_text = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
print(macbeth_text[:500])


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower()


macbeth_text = preprocess_text(macbeth_text)
print(macbeth_text[:500])

macbeth_text_words = (nltk.tokenize.word_tokenize(macbeth_text))
n_words = len(macbeth_text_words)
unique_words = len(set(macbeth_text_words))
print('Total Words: %d' % n_words)
print('Unique Words: %d' % unique_words)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(set(macbeth_text_words)))
tokenizer.fit_on_texts(macbeth_text_words)

vocab_size = len(tokenizer.word_index) + 1
word_2_index = tokenizer.word_index

# Text generation is a many-to-one sequence problem: input is a sequence of words and output is a single word
# LSTM accepts data in 3 dimensions (number of samples, number of time-steps, features per time-step)
# Since output is a single word, is 2 dimensional (number of samples, number of unique words in corpus)

# Modify shape of input sequences and outputs
input_sequence = []
output_words = []
# input sequences will consist of 100 words
input_seq_length = 100

# First iteration int values for the first 100 words from the text are appended to the input_sequence list,
# the 101st word is appended to output_words list.
# Second iteration a sequence of words that starts from the 2nd word in the text and ends at the 101st word is stored
# in the input_sequence list and the 102nd word is stored in the output_words list and so on.
# Total of 17,150 input sequences will be generated since there are 17,250 total words in the dataset
# (100 less than the total words in the text)
for i in range(0, n_words - input_seq_length, 1):
    in_seq = macbeth_text_words[i:i + input_seq_length]
    out_seq = macbeth_text_words[i + input_seq_length]
    input_sequence.append([word_2_index[word] for word in in_seq])
    output_words.append(word_2_index[out_seq])

print(input_sequence[0])

# Let's normalize the input sequences by dividing the ints in the sequences by the largest int value
# and convert output into 2-dim format
X = np.reshape(input_sequence, (len(input_sequence), input_seq_length, 1))
X = X / float(vocab_size)
y = tf.keras.utils.to_categorical(output_words)

print("X shape", X.shape)
print("y shape", y.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(800, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(800, return_sequences=True),
    tf.keras.layers.LSTM(800),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])
print(model.summary())
# Since the output word can be one of 3,436 unique words our problem is a multi-class classification problem.
model.compile(loss='categorical_crossentropy', optimizer='adam')
history = model.fit(X, y, batch_size=64, epochs=10, verbose=1)

# To make predictions we will randomly select a sequence from the input_sequence list, convert it into 3-dimensional
# shape then pass it to the predict method of the trained model.
# The model will return a one-hot encoded array where the index 1 will be the index value of the next word, this index
# is then passed to the index_2_word dictionary where the word index is used as key.

random_seq_index = np.random.randint(0, len(input_sequence) - 1)
random_seq = input_sequence[random_seq_index]

index_2_word = dict(map(reversed, word_2_index.items()))
word_sequence = [index_2_word[value] for value in random_seq]
print(' '.join(word_sequence))

# We created the index_2_word dictionary by reversing the word_2_index dictionary (swapping keys with values)
# Let's print the next 100 words that follow:
for i in range(100):
    int_sample = np.reshape(random_seq, (1, len(random_seq), 1))
    int_sample = int_sample / float(vocab_size)

    predicted_word_index = model.predict(int_sample, verbose=0)

    predicted_word_id = np.argmax(predicted_word_index)
    seq_in = [index_2_word[index] for index in random_seq]

    word_sequence.append(index_2_word[predicted_word_id])

    random_seq.append(predicted_word_id)
    random_seq = random_seq[1:len(random_seq)]

final_output = ""
for word in word_sequence:
    final_output = final_output + " " + word

print(final_output)
