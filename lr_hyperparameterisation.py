from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from pandas import read_csv, DataFrame
from scikit_classifier_funcs import run_model_multiple

"""
Script to tune hyperparameters for the logistic regression classifier. 
Things tuned: 
- c_val
- tol
We then evaluate performance of the classifier that uses the chosen hyperparameters. 
"""


df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

logistic = LogisticRegression(random_state=42, max_iter=1000)

ranked_features = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
features94 = ranked_features.index.values.tolist()[:94]
x_features94 = x[features94]

c_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
tol = [10 ** (-10), 10 ** (-9), 10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** 0]
param_grid = {'C': c_val, 'tol': tol}  # 220 possible combinations

hyper_lr_gridsearch = GridSearchCV(estimator=logistic, param_grid=param_grid)
hyper_lr_gridsearch.fit(x_features94, y)
hyper_lr = DataFrame([hyper_lr_gridsearch.best_params_])
hyper_lr.to_csv('data/hppy_lr_best_hypers.csv', index=False)

hyper_logistic = LogisticRegression(tol=10**(-4), C=1, max_iter=1000, random_state=42)

# hyperparameterised baseline logistic regression - training:test 4:1, full set of features, default hyperparameters
print('Hyper-ed baseline start')
hyper_lr_baseline_stats, hyper_lr_baseline_models = run_model_multiple(hyper_logistic, x_features94, y,
                                                                       test_size=0.2, num_iterations=100,
                                                                       model_type='Logistic Regression',
                                                                       hyper_state='hyper')
hyper_lr_baseline_stats_df = DataFrame([hyper_lr_baseline_stats])
print('Hyper-ed baseline complete')
print(hyper_lr_baseline_stats_df['mean_accuracy'][0])

hyper_lr_baseline_stats_df.to_csv('data/hppy_lr_94features_hyper.csv', index=False)
