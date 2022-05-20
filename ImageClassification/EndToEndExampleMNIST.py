import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def get_uncompiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,), name='digits'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_crossentropy"])
    return model


# Get MNIST as numpy array
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# Preprocess data
X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation (manually)
X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]

model = get_compiled_model()
history = model.fit(X_train, y_train, batch_size=64, epochs=2, validation_data=(X_val, y_val))
print(history.history)

# Let's try with tf.data.Dataset format which can be passed directly between the keras methods
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(64)

# Since the dataset already takes care of batching we don't pass a batch_size argument to fit method.
model.fit(train_dataset, epochs=3)
print("Evaluate")
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))

# Note that the dataset is reset at the end of each epoch, so it can be reused for the next epoch
# If you want to run training only on a specific number of batches from this Dataset, you can pass the steps_per_epoch
# argument, which specifies how many training steps the model should run using this dataset before moving on the next
# epoch.
# If you do this the dataset won't reset at the end of each epoch, instead we just keep drawing the next batches.
# The dataset will eventually run out of data
model.fit(train_dataset, epochs=3, steps_per_epoch=100)

# lets try with validation data using tf.data
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(64)
model.fit(train_dataset, epochs=1, validation_data=val_dataset)

# Using Callbacks
callbacks = [keras.callbacks.EarlyStopping(
    # Stop training when 'val_loss' is no longer improving
    monitor="val_loss",
    # No longer improving being defined as no better than 1e-2 or less
    min_delta=1e-2,
    # No longer improving being further defined as for at least 2 epochs
    patience=2,
    verbose=1)]
model.fit(X_train, y_train, epochs=20, batch_size=64, callbacks=callbacks, validation_split=0.2)

# Checkpointing models with ModelCheckpoint callback
callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,
        monitor="val_loss",
        verbose=1, )]
model.fit(X_train, y_train, epochs=2, batch_size=64, callbacks=callbacks, validation_split=0.2)

# ModelCheckpoint callback can be used to implement fault-tolerance: restart training from the last saved state of the
# model in case training gets randomly interrupted

# Prepare directory to store all checkpoints
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


model = make_or_restore_model()
callbacks = [
    # This callback saves a SavedModel every 100 batches.
    # We include the training loss in the saved model name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", save_freq=100
    )
]
model.fit(X_train, y_train, epochs=1, callbacks=callbacks)


# Using learning rate schedules

# You can use a static learning rate decay schedule by passing a schedule object as the learning_rate argument in
# your optimizer
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96,
                                                          staircase=True)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_crossentropy"])
model.fit(X_train, y_train, epochs=1, callbacks=callbacks)