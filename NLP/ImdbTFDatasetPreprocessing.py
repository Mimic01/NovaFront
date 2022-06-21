import tensorflow as tf
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np

BATCH_SIZE = 64
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

train_ds, ds_info = tfds.load('imdb_reviews', split='train[:80%]', batch_size=BATCH_SIZE, shuffle_files=True,
                              as_supervised=True, with_info=True)
val_ds = tfds.load('imdb_reviews', split='train[80%:]', batch_size=BATCH_SIZE, shuffle_files=True, as_supervised=True)

for review_batch, label_batch in val_ds.take(1):
    for i in range(5):
        print("Review: ", review_batch[i].numpy())
        print("Label: ", label_batch[i].numpy())

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int',
                                                    output_sequence_length=MAX_SEQUENCE_LENGTH)
# Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
train_text = train_ds.map(lambda text, labels: text)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)


def create_model(vocab_size, num_labels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
        tf.keras.layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(num_labels)
    ])
    return model


model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=1)
print(model.summary())
model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))
