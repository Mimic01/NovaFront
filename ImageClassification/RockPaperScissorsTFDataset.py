import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_datasets as tfds

(train_ds, test_ds), ds_info = tfds.load('rock_paper_scissors', split=['train', 'test'], shuffle_files=True,
                                         as_supervised=True,
                                         with_info=True)
for i, data in enumerate(train_ds.take(3)):
    print(i + 1, data[0].shape, data[1])

EPOCHS = 100
NUM_CLASSES = ds_info.features['label'].num_classes
train_size = len(train_ds)
batch_size = 64
img_size = 120


def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (img_size, img_size))
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
        ylabel = "accuracy"
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


train_ds = train_ds.map(normalize_resize, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.map(augment)
train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(normalize_resize, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                     input_shape=(img_size, img_size, 3),
                                                     pooling='max')
x = base_model.output
x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(x)
x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.006),
                          activity_regularizer=tf.keras.regularizers.l1(0.006),
                          bias_regularizer=tf.keras.regularizers.l1(0.006), activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.45, seed=123)(x)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

s = 20 * 2185 // 32
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=s,
                                                               decay_rate=0.1)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
checkpoint_url = 'J:/Novafront/ImageClassification/RPS'
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_url, save_weights_onlt=False, monitor='val_loss',
                                                         save_best_only=True)
callbacks = [early_stopping_cb, model_checkpoint_cb]

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(rate=.2, seed=123),
#     tf.keras.layers.Dense(3)
# ])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=callbacks)

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')
display_training_curves(history.history['categorical_accuracy'],
                        history.history['val_categorical_accuracy'], 'accuracy')
