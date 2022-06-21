import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


def build_model(my_learning_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,))
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate), loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, feature, label, epochs, batch_size, ):
    history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)
    # Gather trained model's weight and bias
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    # List of epochs is stored separately from the rest of history
    epochs = history.epoch
    # Gather the history of each epoch
    hist = pd.DataFrame(history.history)
    # Gather the model's root mean squared error at each epoch
    rmse = hist['root_mean_squared_error']

    return trained_weight, trained_bias, epochs, rmse


def plot_the_model(trained_weight, trained_bias, feature, label):
    plt.xlabel("feature")
    plt.ylabel("label")
    plt.scatter(feature, label)

    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.show()


def plot_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

learning_rate = 0.14
epochs = 70
my_batch_size = 3

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, my_label, epochs, my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_loss_curve(epochs, rmse)


