import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from pandas import read_csv, concat, DataFrame
import numpy as np
from math import floor, ceil, sqrt

""" 
Script to use keras-tuner for hyperparameterisation. Train-val-test splits can be varied.
Things tuned:
- number of dense hidden layers (between 1 and 3)
- number of neurons in each dense hidden layer (between 32 and 512, in increments of 32)
- the activation function of neurons in the dense hidden layers (relu or tanh)
- the presence of a Dropout layer after the hidden layers
- the dropout rate (0.25, 0.5, 0.75)
- the learning rate (between 1e-10 and 1, with log sampling)
"""


df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')


# oversample minority class (inhibitors) in imbalanced data to give a more balanced dataset
# input_x is a df, input_y is a numpy array
def data_balancer(input_x, input_y):
    num_inhibitors = sum(input_y)
    num_non_inhibitors = len(input_y) - num_inhibitors
    if num_non_inhibitors/num_inhibitors < 2:  # data sufficiently balanced
        x_out = input_x
        y_out = input_y
    else:   # data imbalanced
        # x = DataFrame(input_x)
        y = input_y.to_frame()
        data = concat([input_x, y], axis=1)
        times_to_replicate = floor(num_non_inhibitors/num_inhibitors) - 1
        inhibitors = data[data['classification'] == 1]
        inhibitors_replicated = concat([inhibitors]*times_to_replicate, ignore_index=True)
        data_balanced = concat([data, inhibitors_replicated], ignore_index=True)
        x_out = data_balanced.drop(['classification'], axis=1)
        y_out = data_balanced['classification']
    return x_out, y_out


def make_one_dim_array_square(one_d):
    length = len(one_d)
    dim = ceil(sqrt(length))
    square = np.zeros(shape=(dim, dim))
    for i in range(length):
        row_idx = floor(i / dim)
        col_idx = i % dim
        square[row_idx, col_idx] = one_d[i]
    return square


# converts df with fingerprint data into 'images'
def data_df_to_images(df_input):
    df_input_list = [df_input.loc[i] for i in df_input.index]
    output_images = np.array([make_one_dim_array_square(series) for series in df_input_list])
    return output_images


# classification = ndarray
def class_to_logits(classification):
    logits = np.zeros([len(classification), 2])
    for idx, clas in enumerate(classification):
        if clas == 1:
            logits[idx, 1] = 1
        else:
            logits[idx, 0] = 1
    return logits


# function returning a compiled Keras model.
# Takes an argument 'hp' for defining hyperparameters while building the model
# each hyperparameter, e.g. hp.Int(), is defined by a unique name, e.g. hp.Int('num_layers')
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(30, 30)))
    # hyperparameter num_layers tunes the number of hidden layers to have
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),  # hp.Int defines an integer
            # hyperparameter - in this case the hyperparameter covers the number of neurons in this layer
            # f'units_{i} ensures that the number of neurons per layer is tuned independently
            activation=hp.Choice(f'activation_{i}', ['relu', 'tanh'])
            # hyperparameter to choose between activation functions - tuned independently for each layer
        ))
    # hyperparameter 'dropout_rate' to tune dropout_rate to be one of 0.25, 0.5 and 0.75, if a dropout layer is used
    dropout_rate = hp.Float('dropout_rate', min_value=0.25, max_value=0.75, step=0.25)
    if hp.Boolean('dropout'):
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(2))  # standard dense layer with 2 nodes - output layer
    learning_rate = hp.Float('lr', min_value=1e-10, max_value=1, sampling='log')  # hyperparameter to sample learning rates
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  # optimiser - how the model is updated based on the data it sees and the loss func.
                  # Adam = SGD based on adaptive estimation of first- and second-order moments
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # loss function - here, crossentropy loss
                  metrics=['accuracy'])
    return model


train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

callback = EarlyStopping(monitor='val_loss', patience=10)

df_train = df.sample(frac=train_frac+val_frac, random_state=42)
df_test = df.drop(df_train.index)

x_train_df = df_train.drop(['molecule_chembl_id', 'classification'], axis=1)
y_train = df_train['classification']
x_train_df, y_train = data_balancer(x_train_df, y_train)
x_train = data_df_to_images(x_train_df)
y_train = np.array(y_train)
x_test = data_df_to_images(df_test.drop(['molecule_chembl_id', 'classification'], axis=1))
y_test = np.array(df_test['classification'])

# hyperparameter tuning
build_model(kt.HyperParameters())

tuner = kt.Hyperband(
    hypermodel=build_model,
    objective='accuracy',
    max_epochs=500,
    hyperband_iterations=3,
    seed=42,
    overwrite=True
)

tuner.search(x_train, y_train, epochs=500, validation_split=(val_frac / (train_frac + val_frac)), shuffle=False,
             callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)
print(best_hps.space)

tuned_model = build_model(best_hps)

best_hps_df = DataFrame([best_hps.values])
best_hps_df.to_csv('data/hppy_dnn_best_hps_60_20_20_run7.csv', index=False)
