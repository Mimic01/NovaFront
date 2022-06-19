import tensorflow as tf
import tensorflow_datasets as tfds
import plotly.graph_objects as go
import numpy as np

data, ds_info = tfds.load('ag_news_subset', with_info=True, as_supervised=True)
train_ds, test_ds = data['train'], data['test']
for i, data in enumerate(train_ds.take(3)):
    print(i+1, data[0].shape, data[1])

EPOCHS = 20
NUM_CLASSES = ds_info.features["label"].num_classes
train_size = len(train_ds)
batch_size = 64

