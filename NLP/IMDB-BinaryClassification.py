import io
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                          with_info=True, as_supervised=True)
print(info)
# text feature is type=tf.int64 which means reviews have already been tokenized, if you want to read the reviews
# you'll need to decode them
print(info.features)
encoder = info.features['text'].encoder
print(encoder)
string_example = "Mary has a little gun"
print(f"The string example is: {string_example}")
encoded_string = encoder.encode(string_example)
print(f"The encoded string is: {encoded_string}")
original_string = encoder.decode(encoded_string)
print(f"The original string is: {original_string}")

train_data = train_data.shuffle(10000)
val_data = train_data.take(5000)
train_data = train_data.skip(5000)

BUFFER_SIZE = 10000
BATCH_SIZE = 32
# We pad and batch the data in batches of 32, meaning we will feed the model in batches of 32 reviews of the same length
train_batches = train_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
val_batches = val_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE).repeat()
test_batches = test_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
# Remember that tensorflow datasets are composed of batches of data, to access a batch you have to loop on batches
for batch in train_batches:
    print(batch[0])
    print("\n")
    print(batch[1])
    break
# A batch is a tuple composed of 2 elements:
# batch[0] has the 32 reviews of the same length 1317, since it has shape (32, 1317)
# batch[1] has shape (32,), its a 1D vector containing all the labels: ones and zeroes since its a binary classification
# task

# Let's read one review
iterator = train_batches.__iter__()
next_element = iterator.get_next()
one_batch_of_reviews = next_element[0]
one_batch_of_labels = next_element[1]
decoded_review = encoder.decode(one_batch_of_reviews[0])
print(decoded_review)
print("\n")
if one_batch_of_labels[0] == 0:
    print("this person didn't liked the movie")
else:
    print("this person liked this movie")


def plot_graph(history):
    history_dict = history.history

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    fig = plt.figure(figsize=(18, 8))
    plt.subplot2grid((1, 2), (0, 0))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot2grid((1, 2), (0, 1))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# What we need is to have words encoded as tensors (vectors of floating numbers)
embedding_dim = 16
embed_size = 128

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(train_batches, epochs=5, validation_data = test_batches)
plot_graph(history)

