# Predict the output column which can be unacc (unacceptable), acc (acceptable), good or very good
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

sns.set(style="darkgrid")

cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety', 'output']
cars = pd.read_csv(r'J:\Documents\Datahound\Datasets\car-evaluation-dataset\car_evaluation.csv', names=cols,
                   header=None)
print(cars.head())

plot_size = plt.rcParams["figure.figsize"]
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size
cars.output.value_counts().plot(kind='pie', autopct='%0.05f%%', colors=['lightblue', 'lightgreen', 'orange', 'pink'],
                                explode=(0.05, 0.05, 0.05, 0.05))
plt.show()
# Convert categorical columns into numeric columns
price = pd.get_dummies(cars.price, prefix='price')
maint = pd.get_dummies(cars.maint, prefix='maint')
doors = pd.get_dummies(cars.doors, prefix='doors')
persons = pd.get_dummies(cars.persons, prefix='persons')
lug_capacity = pd.get_dummies(cars.lug_capacity, prefix='lug_capacity')
safety = pd.get_dummies(cars.safety, prefix='safety')
labels = pd.get_dummies(cars.output, prefix='condition')
# To create the feature set we merge the first six columns horizontally:
X = pd.concat([price, maint, doors, persons, lug_capacity, safety], axis=1)

print(labels.head())
# Now convert labels into a numpy array since this is the format tensorflow needs for the model
y = labels.values

# Divide the dataset into training and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

EPOCHS = 100
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=8, epochs=EPOCHS, verbose=1, validation_split=0.2)


def display_training_curves(training, validation, yaxis):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
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


display_training_curves(history.history['loss'],history.history['val_loss'], 'loss')

score = model.evaluate(X_valid, y_valid, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
