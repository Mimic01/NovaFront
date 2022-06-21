import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_datasets as tfds

#  Binary Classification with a Structured learning problem

# There are a couple of methods to look at the data. First is to use the as_dataframe() method.
# Second is to iterate over the prefretched data generator object.
de_credit, info = tfds.load('german_credit_numeric', split='train', with_info=True)
de_credit_df = tfds.as_dataframe(de_credit, info)
print(de_credit_df.head())

df = de_credit_df.features.apply(pd.Series)
df.columns = [f"f_{i}" for i in range(df.shape[1])]
df_data = pd.concat([df, de_credit_df.label], axis=1)
print(df_data.shape)
print(df_data.head())
print(info)

# Prepare the dataset
# Split the data intro train and test, then setup the first model
train_len = int(df_data.shape[0] * 0.75)

df_train = df_data[:train_len]
df_val = df_data[train_len:]

X_train = df_train.drop('label', axis=1)
Y_train = df_train.label

X_val = df_val.drop('label', axis=1)
Y_val = df_val.label

print(f"X Train: ", X_train.shape)
print(f"Y Train: ", Y_train.shape)
print(f"X Val: ", X_val.shape)
print(f"Y Val: ", Y_val.shape)


def dnn_model(epochs, callbacks=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, callbacks=callbacks)

    return history, model


hist, model = dnn_model(250)

print(f"Training Set:       {model.evaluate(X_train, Y_train)}")
print(f"Validation Set:     {model.evaluate(X_val, Y_val)}")


def plot(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    hist = history.history

    for ax, metric in zip(axs, ["loss", "accuracy"]):
        ax.plot(hist[metric])
        ax.plot(hist["val_" + metric])
        ax.legend([metric, "val_" + metric])
        ax.set_title(metric)


plot(hist)

# Initial predictions
preds_classes = model.predict(X_val)
pd.Series(preds_classes.flatten()).value_counts().plot(kind='bar', title='Count of Initial Predicted Classes')
X_train.describe()

# Preprocess for better predicting
# Balance dataset with random over sampling
# Scale the dataset with min max scaling
# Apply cross validation

ros = RandomOverSampler(random_state=42)

x_balance, y_balance = ros.fit_resample(df_data.drop("label", axis=1), df_data.label.values)
df_balance = pd.concat([x_balance, pd.DataFrame(y_balance, columns=['label'])], axis=1)

mm = MinMaxScaler()
df_scaled = pd.DataFrame(mm.fit_transform(df_balance), columns=df_balance.columns)
print("Shape after scaling and balancing: ", df_scaled.shape)

train_len = int(df_scaled.shape[0] * 0.75)
df_sample = df_scaled.sample(frac=1)
df_train = df_sample[:train_len]
df_val = df_sample[train_len:]
X_train = df_train.drop('label', axis=1)
Y_train = df_train.label
X_val = df_val.drop('label', axis=1)
Y_val = df_val.label

print(f"X Train: ", X_train.shape)
print(f"Y Train: ", Y_train.shape)
print(f"X Val: ", X_val.shape)
print(f"Y Val: ", Y_val.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
df_train['label'].value_counts().plot(kind='bar', title='Count target label: Training', ax=ax[0])
df_val['label'].value_counts().plot(kind='bar', title="Count target label: Validation", ax=ax[1])


# Regularization
# Early Stopping
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > 0.90:
            print("Accuracy greater than 90%. Stopping training.")
            self.model.stop_training = True


def dnn_model(epochs, callbacks=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, callbacks=callbacks)
    return history, model


hist_regularized, model_regularized = dnn_model(500, callbacks=[EarlyStoppingCallback()])
plot(hist_regularized)

print(f"Training Set:       {model_regularized.evaluate(X_train, Y_train)}")
print(f"Validation Set:     {model_regularized.evaluate(X_val, Y_val)}")

preds_classes = model_regularized.predict(X_val)
pd.Series(preds_classes.flatten()).value_counts().plot(kind='bar', title="Count of Regularized Predicted Classes")

stopvar = 3