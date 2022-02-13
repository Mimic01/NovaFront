import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load('malaria', split='train', shuffle_files=True,
                                         data_dir="J:/Documents/Datahound/Datasets/malaria", as_supervised=True,
                                         with_info=True)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


# Training pipeline
# TFDS provide images of type tf.uint8, while the model expects tf.float32. Therefore, you need to normalize images.
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# As you fit the dataset in memory, cache it before shuffling for a better performance.
ds_train = ds_train.cache()
# For true randomness, set the shuffle buffer to the full dataset size.
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# Batch elements of the dataset after shuffling to get unique batches at each epoch.
ds_train = ds_train.batch(128)
# It is good practice to end the pipeline by prefetching
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Evaluation pipeline
# You don't need to call tf.data.Dataset.shuffle.
# Caching is done after batching because batches can be the same between epochs.
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
