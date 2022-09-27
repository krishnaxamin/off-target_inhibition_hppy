from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pandas import DataFrame, Series, concat
from math import floor
from statistics import mean
from random import seed
from random import randint

"""
Collection of functions used wherever scikit-learn regressors are involved elsewhere in the project.
"""


# Function to train an initialised but unfitted regression model multiple times from scratch.
# Uses train_test_split to randomly split dataset into training and test multiple times.
# Based off run_model_multiple (built for classifiers)
# Outputs tuple: dictionary of model info and performance; average feature_importances_ (regression coefficients for
# each feature)
def run_regressor_multiple(modeller, x_whole, y_whole,
                           test_size=0.2,
                           num_iterations=10,
                           model_type='Random Forest',
                           hyper_state='default',
                           verbose=True,
                           track_individual_models=True,
                           track_features=False):
    num_features = x_whole.shape[1]
    features_names = list(x_whole.columns.values)
    r2_list = []
    rmse_list = []
    mae_list = []
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
        modeller.fit(x_train_multiple, y_train_multiple)

        # calculate R-squared (r2)
        r2 = modeller.score(x_test_multiple, y_test_multiple)

        # calculate root mean squared error (RMSE)
        y_pred_multiple = modeller.predict(x_test_multiple)
        rmse = mean_squared_error(y_test_multiple, y_pred_multiple, squared=False)

        # calculate mean absolute error (MAE)
        mae = mean_absolute_error(y_test_multiple, y_pred_multiple)

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
            print('Iteration R2: ' + str(r2))
            print('Iteration RMSE: ' + str(rmse))
            print('Iteration MAE: ' + str(mae))
        if track_individual_models:
            individual_models.append([counter, random_state, r2, rmse, mae])
        counter += 1
        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)

    individual_models_df = DataFrame(individual_models, columns=['model', 'random_state', 'r2', 'rmse',
                                                                 'mae'])
    if track_features:
        features_df = features_df.reset_index(drop=True)
        individual_models_df = concat([individual_models_df, features_df], axis=1)
    return [{'model_type': model_type, 'test_fraction': test_size, 'num_features': num_features,
            'hyperparameter_state': hyper_state, 'num_iterations': num_iterations, 'mean_r2': mean(r2_list),
             'mean_rmse': mean(rmse_list), 'mean_mae': mean(mae_list)},
            individual_models_df]
