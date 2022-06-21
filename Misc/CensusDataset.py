import tensorflow as tf
import pandas as pd

COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship',
           'race', 'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country', 'label']
PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
PATH_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(PATH, skipinitialspace=True, names=COLUMNS, index_col=False)
df_test = pd.read_csv(PATH_test, skiprows=1, skipinitialspace=True, names=COLUMNS, index_col=False)

print(df_train.shape, df_test.shape)
print(df_train.dtypes)

# Tf requires a boolean to train the classifier, you need to cast the values from string to int.
# the label is stored as an object o you need to convert it into a numeric value, this code creates
# a dictionary with the values to convert and loop over the column item.

label = {'<=50K': 0, '>50K': 1}
df_train.label = [label[item] for item in df_train.label]

label_t = {'<=50K.': 0, '>50K.': 1}
df_test.label = [label_t[item] for item in df_test.label]

print(df_train["label"].value_counts())
print(df_test["label"].value_counts())
print(df_train.dtypes)

CONTI_FEATURES = ['age', 'fnlwgt', 'capital_gain', 'education_num', 'capital_loss', 'hours_week']
CATE_FEATURES = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'native_country']

continuous_features = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES]
categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size=1000) for k in
                        CATE_FEATURES]

model = tf.estimator.LinearClassifier(
    n_classes=2,
    model_dir="ongoing/train",
    feature_columns=categorical_features + continuous_features)

FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race',
            'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country']
LABEL = 'label'


def get_input_fn(data_set, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
                                               y=pd.Series(data_set[LABEL].values),
                                               batch_size=n_batch,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle)


model.train(input_fn=get_input_fn(df_train, num_epochs=None, n_batch=128, shuffle=False, steps=1000))
model.evaluate(input_fn=get_input_fn(df_test,
                                      num_epochs=1,
                                      n_batch = 128,
                                      shuffle=False),
                                      steps=1000)