from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from pandas import DataFrame, Series, concat
from math import floor
from statistics import mean
from random import seed
from random import randint

"""
Collection of functions used wherever scikit-learn classifiers are involved elsewhere in the project.
"""


# oversample minority class (inhibitors) in imbalanced data to give a more balanced dataset
# input_x is a df, input_y is a numpy array
def data_balancer(input_x, input_y):
    num_inhibitors = sum(input_y)
    num_non_inhibitors = len(input_y) - num_inhibitors
    # data sufficiently balanced
    if num_non_inhibitors/num_inhibitors < 2:
        x_out = input_x
        y_out = input_y

    # data imbalanced
    else:
        y = input_y.to_frame()
        data = concat([input_x, y], axis=1)
        times_to_replicate = floor(num_non_inhibitors/num_inhibitors) - 1
        inhibitors = data[data['classification'] == 1]
        inhibitors_replicated = concat([inhibitors]*times_to_replicate, ignore_index=True)
        data_balanced = concat([data, inhibitors_replicated], ignore_index=True)
        x_out = data_balanced.drop(['classification'], axis=1)
        y_out = data_balanced['classification']
    return x_out, y_out


# Function to train an initialised but unfitted model multiple times from scratch.
# Uses train_test_split to randomly split dataset into training and test multiple times.
# Outputs tuple: dictionary of model info and performance; average feature_importances_ (regression coefficients for
# each feature)
def run_model_multiple(modeller, x_whole, y_whole,
                       test_size=0.2,
                       num_iterations=10,
                       model_type='Random Forest',
                       hyper_state='default',
                       balance_data=False,
                       verbose=True,
                       track_individual_models=True,
                       track_features=False):
    num_features = x_whole.shape[1]
    features_names = list(x_whole.columns.values)
    accuracy_list = []
    f1_list = []
    sensitivity_list = []
    specificity_list = []
    balanced_accuracy_list = []
    counter = 1
    features_df = DataFrame()
    individual_models = []
    seed(42)  # initialises seed for random numbers, so that the same set of random numbers is generated each time
    # objective: random coverage of all space, but replicable between different models and thus comparable
    for itera in range(num_iterations):
        if verbose:
            print(counter)
        random_state = randint(1, 1000)
        if verbose:
            print('Seed: ' + str(random_state))
        x_train_multiple, x_test_multiple, y_train_multiple, y_test_multiple = train_test_split(x_whole, y_whole,
                                                                                                test_size=test_size,
                                                                                                random_state=random_state)
        if balance_data:
            x_train_balanced, y_train_balanced = data_balancer(x_train_multiple, y_train_multiple)
            modeller.fit(x_train_balanced, y_train_balanced)
        else:
            modeller.fit(x_train_multiple, y_train_multiple)

        # calculate accuracy and f1 score
        accuracy = modeller.score(x_test_multiple, y_test_multiple)
        f1 = f1_score(y_test_multiple, modeller.predict(x_test_multiple))

        # calculate sensitivity and specificity
        y_pred_multiple = modeller.predict(x_test_multiple)
        confusion_mat = confusion_matrix(y_test_multiple, y_pred_multiple)  # confusion matrix
        true_negatives = confusion_mat[0, 0]
        false_negatives = confusion_mat[1, 0]
        true_positives = confusion_mat[1, 1]
        false_positives = confusion_mat[0, 1]
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)

        balanced_accuracy = mean([sensitivity, specificity])  # calculate balanced accuracy
        if model_type != 'Random Forest':
            if model_type == 'Decision Tree':
                features = DataFrame([Series(modeller.feature_importances_)])
                new_colname_list = []
                for colname in list(features_names):
                    new_colname_list.append(str(colname) + '_gini/coefs')
                features.columns = new_colname_list
                features_df = concat([features_df, features])
            elif model_type == 'Logistic Regression':
                features = DataFrame([Series(modeller.coef_.flatten())])
                new_colname_list = []
                for colname in list(features_names):
                    new_colname_list.append(str(colname) + '_gini/coefs')
                features.columns = new_colname_list  # here
                features_df = concat([features_df, features])
        else:
            features = DataFrame([Series(modeller.feature_importances_)])
            new_colname_list = []
            for colname in list(features_names):
                new_colname_list.append(str(colname) + '_gini/coefs')
            features.columns = new_colname_list
            features_df = concat([features_df, features])
        if verbose:
            print('Iteration accuracy: ' + str(accuracy))
            print('Iteration sensitivity: ' + str(sensitivity))
            print('Iteration specificity: ' + str(specificity))
            print('Iteration balanced accuracy: ' + str(balanced_accuracy))
            print('Iteration F1: ' + str(f1))
        if track_individual_models:
            individual_models.append([counter, random_state, accuracy, sensitivity, specificity, balanced_accuracy, f1])
        counter += 1
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        balanced_accuracy_list.append(balanced_accuracy)
        f1_list.append(f1)

    individual_models_df = DataFrame(individual_models, columns=['model', 'random_state', 'accuracy', 'sensitivity',
                                                                 'specificity', 'balanced_accuracy', 'f1'])
    if track_features:
        features_df = features_df.reset_index(drop=True)
        individual_models_df = concat([individual_models_df, features_df], axis=1)
    return [{'model_type': model_type, 'test_fraction': test_size, 'num_features': num_features,
            'hyperparameter_state': hyper_state, 'num_iterations': num_iterations, 'mean_accuracy': mean(accuracy_list),
             'mean_sensitivity': mean(sensitivity_list), 'mean_specificity': mean(specificity_list),
             'mean_balanced_accuracy': mean(balanced_accuracy_list), 'mean_f1': mean(f1_list)}, individual_models_df]


# PubchemFP0_coeff -> PubchemFP0
def pubchem_renamer(input_list):
    output_list = []
    for name in input_list:
        new_name = name.split('_')[0]
        output_list.append(new_name)
    return output_list
