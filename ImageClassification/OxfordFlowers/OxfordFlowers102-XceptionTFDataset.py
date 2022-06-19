import tensorflow as tf
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np

# as_supervised
# bool, if True, the returned tf.data.Dataset will have a 2-tuple structure (input, label)
# f False, the default, the returned tf.data.Dataset will have a dictionary with all the features
data, ds_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
train_ds, valid_ds, test_ds = data['train'], data['validation'], data['test']
for i, data in enumerate(train_ds.take(3)):
    print(i+1, data[0].shape, data[1])

EPOCHS = 20
NUM_CLASSES = ds_info.features["label"].num_classes
train_size = len(train_ds)
batch_size = 64
img_size = 120


def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (img_size, img_size))
    # int label to one hot encoded label
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


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


train = train_ds.map(normalize_resize).cache().map(augment).shuffle(100).batch(batch_size).repeat()
valid = valid_ds.map(normalize_resize).cache().batch(batch_size)
test = test_ds.map(normalize_resize).cache().batch(batch_size)

base_model = tf.keras.applications.Xception(weights='imagenet', input_shape=(img_size, img_size, 3),
                                            include_top=False)
base_model.trainable = False
inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(train, steps_per_epoch=train_size // batch_size, epochs=EPOCHS, validation_data=valid, verbose=2)

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')
display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                        'accuracy')