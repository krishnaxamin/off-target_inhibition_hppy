import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Softmax, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Accuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from pandas import read_csv, concat, DataFrame
import numpy as np
from math import floor, ceil, sqrt
from statistics import mean

""" 
Script to run tuned models, get their losses on training and validation data over their training, their 
performance on test data and their predictions on query UKB data.
"""

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')


# oversample minority class (inhibitors) in imbalanced data to give a more balanced dataset
# input_x is a df, input_y is a numpy array
# will only balance data if it needs to be balanced
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


def build_tuned_model():
    best_hps_df = read_csv('data/hppy_dnn_best_hps_' + which_hps + '.csv')
    model = tf.keras.Sequential([
        Flatten(input_shape=(30, 30)),
        Dense(160, activation='relu'),
        Dropout(rate=0.25),
        Dense(2)])
    learning_rate = best_hps_df['lr'][0]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')
    return model


# sets up a neural forest with num_nets neural nets and trains the nets within the forest.
# Gets train-val histories (to track accuracies and losses over epochs) as a num_nets-long list of
# keras.callbacks.History objects, stats on test data and predictions on query data.
def neural_forest(num_nets, data_input, callback_forest, train_frac_forest=0.8, val_frac_forest=0.1,
                  num_epochs=50, shuffle_bool=False):
    assert len(data_input) == 4, 'Number of data inputs should be 4 - x_train, y_train, x_test and x_predict'
    forest_train_val_stats = DataFrame()
    forest_test_votes = DataFrame()
    forest_predict_votes = DataFrame()
    histories = []
    x_train_forest = data_input[0]
    y_train_forest = data_input[1]
    for i in range(num_nets):
        print(i + 1)
        model = build_tuned_model()
        history = model.fit(x_train_forest, y_train_forest, epochs=num_epochs,
                            validation_split=(val_frac_forest / (train_frac_forest + val_frac_forest)),
                            shuffle=shuffle_bool, callbacks=[callback_forest])
        histories.append(history)

        # get train-val stats
        history_df = DataFrame(history.history)
        train_val_stats = history_df.iloc[[-1]]
        forest_train_val_stats = concat([forest_train_val_stats, train_val_stats])

        prediction_model = tf.keras.Sequential([model, Softmax()])  # modify net to make predictions
        # get test predictions
        x_test_forest = data_input[2]
        test_prediction_probabilities = prediction_model.predict(x_test_forest)
        test_prediction = np.array(
            [np.argmax(test_prediction_probabilities[i]) for i in range(test_prediction_probabilities.shape[0])])
        forest_test_votes = concat([forest_test_votes, DataFrame([test_prediction])])

        # get UKB predictions
        x_predict_forest = data_input[3]
        prediction_probabilities = prediction_model.predict(x_predict_forest)
        prediction = np.array(
            [np.argmax(prediction_probabilities[i]) for i in range(prediction_probabilities.shape[0])])
        forest_predict_votes = concat([forest_predict_votes, DataFrame([prediction])])

    forest_test_consensus = forest_test_votes.mean()
    forest_test_consensus_out = np.round(np.array(forest_test_consensus))
    forest_predict_consensus = forest_predict_votes.mean()
    forest_predict_consensus_out = np.round(np.array(forest_predict_consensus))

    return [histories, forest_test_consensus_out, forest_predict_consensus_out]


# function to generate performance values, given a set of predictions and a set of target values
def performance_metrics(y_true, y_pred):
    accuracy = Accuracy()
    accuracy.update_state(y_true, y_pred)
    accuracy_val = accuracy.result().numpy()
    fn = FalseNegatives()
    fn.update_state(y_true, y_pred)
    fn_val = fn.result().numpy()
    fp = FalsePositives()
    fp.update_state(y_true, y_pred)
    fp_val = fp.result().numpy()
    tn = TrueNegatives()
    tn.update_state(y_true, y_pred)
    tn_val = tn.result().numpy()
    tp = TruePositives()
    tp.update_state(y_true, y_pred)
    tp_val = tp.result().numpy()
    print(tp_val)
    print(fp_val)
    print(tn_val)
    print(fn_val)
    sensitivity = tp_val / (tp_val + fn_val)  # same as recall
    specificity = tn_val / (tn_val + fp_val)
    balanced_accuracy = mean([sensitivity, specificity])
    precision = tp_val / (tp_val + fp_val)
    f1 = 2 * (sensitivity * precision)/(sensitivity + precision)
    return {'accuracy': accuracy_val, 'sensitivity': sensitivity, 'specificity': specificity,
            'balanced_accuracy': balanced_accuracy, 'f1': f1}


# fractions of the whole dataset to be used for training, validation and testing
train_frac = 0.6
val_frac = 0.2
test_frac = 0.2
which_hps = '60_20_20_run4'
model_version_as_tunedx = 'tuned3'
run_num = 'run1'
file_prefix = 'dnn_' + model_version_as_tunedx + '_' + run_num

callback = EarlyStopping(monitor='val_loss', patience=10)  # Early stopping while monitoring loss on validation data

# Get train and test data
df_train = df.sample(frac=train_frac+val_frac, random_state=42)
df_test = df.drop(df_train.index)

x_train_df = df_train.drop(['molecule_chembl_id', 'classification'], axis=1)
y_train = df_train['classification']
x_train_df, y_train = data_balancer(x_train_df, y_train)  # balance the training data only

x_train = data_df_to_images(x_train_df)
y_train = np.array(y_train)
x_test = data_df_to_images(df_test.drop(['molecule_chembl_id', 'classification'], axis=1))
y_test = np.array(df_test['classification'])

# Get query data - UKB data
ukb_drugs_descriptor = read_csv('data/drug_ukb_name_fingerprints.csv')
ukb_drugs_fingerprints = ukb_drugs_descriptor.drop(['Name'], axis=1)
ukb_drugs_notna = read_csv('data/drug_ukb_cleaned.csv')
ukb_drugs_images = data_df_to_images(ukb_drugs_fingerprints)

# run the forest
forest = neural_forest(100, [x_train, y_train, x_test, ukb_drugs_images],
                       callback_forest=callback,
                       train_frac_forest=train_frac,
                       val_frac_forest=val_frac,
                       num_epochs=500)

# unpack the forest output
forest_histories = forest[0]
# forest_train_val = forest[1]
forest_test = forest[1]
forest_predict = forest[2]

# save histories
for i, history in enumerate(forest_histories):
    history_df = DataFrame(history.history)
    history_df.to_csv(f'data/dnn_' + model_version_as_tunedx + '/' + run_num + '/' + file_prefix + '_training_val_history_{i}.csv',
                      index=False)

# forest_train_val.to_csv('data/dnn_tuned1/run1/dnn_tuned1_run1_train_val_stats.csv', index=False)

# save stats for performance on test data
stats_testing = performance_metrics(y_true=y_test, y_pred=forest_test)
DataFrame([stats_testing]).to_csv('data/dnn_' + model_version_as_tunedx + '/' + run_num + '/' + file_prefix + '_test_stats.csv',
                                  index=False)

# save predictions on UKB data
ukb_drugs_dnn_classed = concat([ukb_drugs_notna.drop(['Drug', 'Drug_curated', 'smiles'], axis=1),
                                DataFrame(np.vstack(forest_predict), columns=['predicted_classification'])], axis=1)
dnn_active_ukb_drugs = ukb_drugs_dnn_classed[ukb_drugs_dnn_classed['predicted_classification'] == 1]
dnn_active_ukb_drugs.to_csv('data/dnn_' + model_version_as_tunedx + '/' + run_num + '/' + file_prefix + '_active_ukb_drugs.csv', index=False)