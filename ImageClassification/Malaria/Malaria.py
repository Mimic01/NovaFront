import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test, ds_valid), ds_info = tfds.load('malaria', split=['train', 'test', 'validation'], shuffle_files=True,
                                          data_dir="J:/Documents/Datahound/Datasets/oxford_flowers102",
                                          as_supervised=True,
                                          with_info=True)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train']).num_examples

