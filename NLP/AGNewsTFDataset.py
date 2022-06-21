import tensorflow as tf
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np

BATCH_SIZE = 64
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

data, ds_info = tfds.load('ag_news_subset', with_info=True, batch_size=BATCH_SIZE, shuffle_files=True,
                          as_supervised=True)
train_ds, test_ds = data['train'], data['test']
for i, data in enumerate(train_ds.take(3)):
    print(i + 1, data[0].shape, data[1])

for review_batch, label_batch in test_ds.take(1):
    for i in range(5):
        print("Review: ", review_batch[i].numpy())
        print("Label: ", label_batch[i].numpy())

EPOCHS = 20
NUM_CLASSES = ds_info.features["label"].num_classes
train_size = len(train_ds)
batch_size = 64

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int',
                                                    output_sequence_length=MAX_SEQUENCE_LENGTH)
train_text = train_ds.map(lambda text, labels: text)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = train_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


train_ds = configure_dataset(train_ds)
test_ds = configure_dataset(test_ds)


def create_model(vocab_size, num_labels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
        tf.keras.layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(num_labels)
    ])
    return model


model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=NUM_CLASSES)
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
              metrics=['sparse_categorical_accuracy'])
print(model.summary())
history = model.fit(train_ds, validation_data=test_ds, epochs=3)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))
