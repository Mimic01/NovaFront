import tensorflow as tf
import tensorflow_datasets as tfds

# as_supervised=True: Returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}.
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True,
                                         with_info=True)
print(ds_info.features)
print(ds_info.features.shape)
print(ds_info.features.dtype)
print(ds_info.features['image'].shape)
print(ds_info.features['image'].dtype)
print(ds_info.splits)
# available splits
print(list(ds_info.splits.keys()))
# Get info on individual splits
print(ds_info.splits['train'].num_examples)
print(ds_info.splits['train'].filenames)
print(ds_info.splits['train'].num_shards)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


# TFDS provides images of type tf.uint8, while the model expects tf.float32. Therefore you need to normalize the images.
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# As you fit the dataset in memory, cache it before shuffling for a better performance
# Random transformations should be applied after caching
ds_train = ds_train.cache()
# For true randomness, set the shuffle buffer to the full dataset size
# For large datasets that can't fit into memory, use buffer_size=1000 if your system allows it.
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# Batch elements of the dataset after shuffling to get unique batches at each epoch
ds_train = ds_train.batch(128)
# It is good practice to end the pipeline by prefetching for performance
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# The testing pipeline is similar but you don't need to shuffle and caching is done after batching because batches can
# be the same between epochs.
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(ds_train, epochs=6, validation_data=ds_test)
