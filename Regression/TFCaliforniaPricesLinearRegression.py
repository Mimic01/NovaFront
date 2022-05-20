import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets"
                                             "/california_housing_train.csv")
# Scale the label (the column we want to predict)
# It will put the value of each house in units of thousands, will keep loss values and learning rates in a
# frendlier range
training_df["median_house_value"] /= 1000.0
print(training_df.head())
print(training_df.describe())


def build_model(my_learning_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,))
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate), loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, df, feature, label, epochs, batch_size):
    history = model.fit(x=df[feature], y=df[label], batch_size=batch_size, epochs=epochs)
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse


def plot_model(trained_weight, trained_bias, feature, label):
    plt.xlabel(feature)
    plt.ylabel(label)
    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])
    # Create red line representing the model
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.show()


def plot_loss(epochs, rmse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


def predict_house_values(n, feature, label):
    batch = training_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)
    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                      training_df[label][10000 + i],
                                      predicted_values[i][0]))


# Importantly you have to define which features correlate with the label (prediction)
# Here you'll arbitrarily use total_rooms as a feature
learning_rate = 0.01
epochs = 30
batch_size = 30
my_feature = "total_rooms"
my_label = "median_house_value"
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df, my_feature, my_label, epochs, batch_size)
print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)
plot_model(weight, bias, my_feature, my_label)
plot_loss(epochs, rmse)
predict_house_values(10, my_feature, my_label)

# Lets try with the 'population' feature
my_feature = "population"
learning_rate = 0.05
epochs = 18
batch_size = 3
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

plot_model(weight, bias, my_feature, my_label)
plot_loss(epochs, rmse)
predict_house_values(10, my_feature, my_label)

# Let's define a synthetic feature, maybe tha ratio of total_rooms to population might have predictive power
# This makes better predictions but it still lacking better accuracy
training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]
my_feature = "rooms_per_person"
learning_rate = 0.06
epochs = 24
batch_size = 30
my_model = build_model(learning_rate)
weight, bias, epochs, mae = train_model(my_model, training_df, my_feature, my_label, epochs, batch_size)
plot_loss(epochs, mae)
predict_house_values(15, my_feature, my_label)

# Find features whose raw values correlate with the label using a correlation matrix
# 1.0 perfect positive correlation, when one attribute rises the other also rises
# -1.0 perfect negative correlation, when one attribute rises the other one falls
# 0.0 no correlation
# In general the higher the absolute value of a correlation value, the greater its predictive power,
# for instance a correlation of -0.8 implies more predictive power than -0.2

# Generate correlation matrix
training_df.corr()
# median_income seems to correlate well with a value of 0.7