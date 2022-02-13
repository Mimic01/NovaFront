# Download the Bach chorales dataset and unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach.
# Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to
# a note's index on a piano (except for the value 0, which means that no note is played). Train a model—recurrent,
# convolutional, or both—that can predict the next time step (four notes), given a sequence of time steps
# from a chorale. Then use this model to generate Bach-like music, one note at a time: you can do this by giving the
# model the start of a chorale and asking it to predict the next time step, then appending these time steps to the
# input sequence and asking the model for the next note, and so on. Also make sure to check out Google's Coconet model,
# which was used for a nice Google doodle about Bach.

import sklearn
import tensorflow as tf
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import pandas as pd

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/"
FILENAME = "jsb_chorales.tgz"
filepath = keras.utils.get_file(FILENAME,
                                DOWNLOAD_ROOT + FILENAME,
                                cache_subdir="datasets/jsb_chorales",
                                extract=True)

jsb_chorales_dir = Path(filepath).parent
train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))


def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]


train_chorales = load_chorales(train_files)
valid_chorales = load_chorales(valid_files)
test_chorales = load_chorales(test_files)

train_chorales[0]

# Notes range from 36 (C1 = C on octave 1) to 81 (A5 = A on octave 5), plus 0 for silence

notes = set()
for chorales in (train_chorales, valid_chorales, test_chorales):
    for chorale in chorales:
        for chord in chorale:
            notes |= set(chord)

n_notes = len(notes)
min_note = min(notes - {0})
max_note = max(notes)

assert min_note == 36
assert max_note == 81

from IPython.display import Audio
from IPython.display import display


def notes_to_frequencies(notes):
    # Frequency doubles when you go up one octave; there are 12 semi-tones
    # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
    return 2 ** ((np.array(notes) - 69) / 12) * 440


def frequencies_to_samples(frequencies, tempo, sample_rate):
    note_duration = 60 / tempo  # the tempo is measured in beats per minutes
    # To reduce click sound at every beat, we round the frequencies to try to
    # get the samples close to zero at the end of each note.
    frequencies = np.round(note_duration * frequencies) / note_duration
    n_samples = int(note_duration * sample_rate)
    time = np.linspace(0, note_duration, n_samples)
    sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
    sine_waves *= (frequencies > 9.).reshape(-1, 1)
    return sine_waves.reshape(-1)


def chords_to_samples(chords, tempo, sample_rate):
    freqs = notes_to_frequencies(chords)
    freqs = np.r_[freqs, freqs[-1:]]  # make last note a bit longer
    merged = np.mean([frequencies_to_samples(melody, tempo, sample_rate)
                      for melody in freqs.T], axis=0)
    n_fade_out_samples = sample_rate * 60 // tempo  # fade out last note
    fade_out = np.linspace(1., 0., n_fade_out_samples) ** 2
    merged[-n_fade_out_samples:] *= fade_out
    return merged


def play_chords(chords, tempo=160, amplitude=0.1, sample_rate=44100, filepath=None):
    samples = amplitude * chords_to_samples(chords, tempo, sample_rate)
    if filepath:
        from scipy.io import wavfile
        samples = (2 ** 15 * samples).astype(np.int16)
        wavfile.write(filepath, sample_rate, samples)
        return display(Audio(filepath))
    else:
        return display(Audio(samples, rate=sample_rate))


# Listen to a few chorales
for index in range(3):
    play_chords(train_chorales[index])


#  It's much better and simpler to predict one note at a time
#  We will use a sequence-to-sequence approach, where we feed a window to the neural net, and it tries to predict
#  that same window shifted one time step into the future.
# We will also shift the values so that they range from 0 to 46, where 0 represents silence, and values 1 to 46
# represent notes 36 (C1) to 81 (A5).
# And we will train the model on windows of 128 notes (i.e., 32 chords).
# All the preprocessing using tf.data

def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:]
    return X, Y


def preprocess(window):
    window = tf.where(window == 0, window, window - min_note + 1)  # shift values
    return tf.reshape(window, [-1])  # convert to arpegio


def bach_dataset(chorales, batch_size=32, shuffle_buffer_size=None,
                 window_size=32, window_shift=16, cache=True):
    def batch_window(window):
        return window.batch(window_size + 1)

    def to_windows(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_windows).map(preprocess)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)
    return dataset.prefetch(1)


train_set = bach_dataset(train_chorales, shuffle_buffer_size=1000)
valid_set = bach_dataset(valid_chorales)
test_set = bach_dataset(test_chorales)

n_embedding_dims = 5

model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_notes, output_dim=n_embedding_dims,
                           input_shape=[None]),
    keras.layers.Conv1D(32, kernel_size=2, padding="causal", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(48, kernel_size=2, padding="causal", activation="relu", dilation_rate=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu", dilation_rate=4),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=8),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.Dense(n_notes, activation="softmax")
])
model.summary()

optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(train_set, epochs=20, validation_data=valid_set)

model.save("my_bach_model.h5")
model.evaluate(test_set)


# Now let's write a function that will generate a new chorale. We will give it a few seed chords,
# it will convert them to arpegios (the format expected by the model), and use the model to predict the next note,
# then the next, and so on. In the end, it will group the notes 4 by 4 to create chords again,
# and return the resulting chorale.

def generate_chorale(model, seed_chords, length):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            # next_note = model.predict_classes(arpegio)[:1, -1:]
            next_note = np.argmax(model.predict(arpegio), axis=-1)[:1, -1:]
            arpegio = tf.concat([arpegio, next_note], axis=1)
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])


seed_chords = test_chorales[2][:8]
play_chords(seed_chords, amplitude=0.2)

new_chorale = generate_chorale(model, seed_chords, 56)
play_chords(new_chorale)


def generate_chorale_v2(model, seed_chords, length, temperature=1):
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            next_note_probas = model.predict(arpegio)[0, -1:]
            rescaled_logits = tf.math.log(next_note_probas) / temperature
            next_note = tf.random.categorical(rescaled_logits, num_samples=1)
            arpegio = tf.concat([arpegio, next_note], axis=1)
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])


new_chorale_v2_cold = generate_chorale_v2(model, seed_chords, 56, temperature=0.8)
play_chords(new_chorale_v2_cold, filepath="bach_cold.wav")

new_chorale_v2_medium = generate_chorale_v2(model, seed_chords, 56, temperature=1.0)
play_chords(new_chorale_v2_medium, filepath="bach_medium.wav")

new_chorale_v2_hot = generate_chorale_v2(model, seed_chords, 56, temperature=1.5)
play_chords(new_chorale_v2_hot, filepath="bach_hot.wav")