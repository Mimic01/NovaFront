import tensorflow as tf
import numpy as np
import os
import time

print('Loading the data...')
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
print('...Data loaded.')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(text[0:250])

# set selects the unique characters
vocab = sorted(set(text))
print(f"There are {format(len(vocab))} unique characters in the whole text.")

# let's tokenize, we'll create 2 lookup tables: one to transform characters to numbers and one that goes reverse
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
# array that contains all the characters as numbers
text_as_int = np.array([char2idx[c] for c in text])

# Since we want our model to generate text, we need to train it to predict the character following a given sequence.
# We will make it train on sequence of 99 characters. At each character, the model will try to predict the next one
# until it reaches the 99th and tries to predict the 100th. It will then move to the next sequence and do the same.

# the next thing we have to do is to split the data into
# len(text_as_int)//(seq_length+1) parts and put it into a TensorFow Dataset

# numbers of characters in one sequence of words, for now it contains the input and the target since it's size is 100
seq_length = 100
examples_per_epoch = text_as_int.shape[0] // (seq_length + 1)

# Creating a TF Dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Now we want sequences of length = seq_length so we concatenate with batching
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


# Remember, we want the model to predict the next character after a sequence of 99 letters while training on
# predicting each letter of the sequence. As the result, the target just have to be the same sequence, but shifted from
# one character to the right.
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
# Before we batched to get the sequences, now we batch to make packs of inputs for feeding the model
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# We got 2 batches of 64 sequences with 100 letters
print(dataset)

vocab_size = len(vocab)
# Each character will be encoded in a vector of dimension of embedding_dim
embedding_dim = 256
rnn_units = 1024


# This is a categorical classification problem since you want to guess the letter given len(vocab_size) characters
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
    # Let's predict the input_example_batch
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# 64 results since the batch is 64 long
# 100 predicted characters per sequence
# 65  is the probability of getting each character

print(model.summary())
# tf.random.categorical(logits, num_samples) logits is a 2D float tensor with shape [batch_size, number_of_classes]
# each row of the logit tensor represents the event probabilities of a different categorical distribution
# num_samples is the number of samples we want to select
samples_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
print(samples_indices)
samples_indices = tf.squeeze(samples_indices, axis=-1).numpy()
# Now it's an array of 100 elements. This is the prediction for the first sequence of the batch
print(samples_indices)


# We'll use sparse_categorical_crossentropy since the problem is categorical
# Since the model gives log of probabilities, we have to flag the from_logits
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS = 20
# There's no need to specify the input and output data when the dataset is constructed with tuples containing two
# elements, the input and the output.
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Generating text sequences

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
print(model.summary())


def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 0.5

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)


print(generate_text(model, start_string=u"ROMEO: "))
