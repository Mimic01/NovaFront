import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import tensorflow_hub as hub

train_data, test_data = tfds.load(name='imdb_reviews', split=['train', 'test'], batch_size=-1, as_supervised=True)
train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

EPOCHS = 40

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
print(train_examples[:10])
print(train_labels[:10])

# One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text
# embedding as the first layer, which will have two advantages:
# we don't have to worry about text preprocessing,
# we can benefit from transfer learning.
model = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
print(hub_layer(train_examples[:3]))

model = tf.keras.models.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
print(model.summary())

# The model that we are using (google/nnlm-en-dim50/2) splits the sentence into tokens, embeds each token and then
# combines the embedding. The resulting dimensions are: (num_examples, embedding_dimension).
model.compile(optimizer="adam", loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

x_val = train_examples[:10000]
partial_x_train = train_examples[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=EPOCHS, batch_size=512, validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_examples, test_labels)
print(results)


def display_training_curves(training, validation, yaxis):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS + 1), mode='lines+markers', y=training, marker=dict(color="dodgerblue"),
                   name='Train'))
    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS + 1), mode='lines+markers', y=validation, marker=dict(color='darkorange'),
                   name="Val"))
    fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs", template="plotly_white")
    fig.show()


display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy')
