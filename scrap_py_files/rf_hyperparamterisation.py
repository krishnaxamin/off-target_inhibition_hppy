from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from pandas import read_csv, DataFrame
import numpy as np
from scikit_classifier_funcs import run_model_multiple

"""
Script to tune hyperparameters for the random forest classifier. 
Things tuned: 
- the number of estimators in the forest (between 50 and 5000, with a step of 20)
- the number of features to consider at every split (between auto and sqrt)
- the maximum number of levels in a tree (between 10 and 100, with a step of 10)
- the minimum number of samples required to split a node (2, 4, 8 or 16)
- the minimum number of samples required at each leaf node (1, 2, 4 or 8)
- whether you use bootstrap to select samples for training each tree
We then evaluate performance of the classifier that uses the chosen hyperparameters while using a train-test split of 
either 80-20 or 90-10. 
"""

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

randomforest = RandomForestClassifier(random_state=42)
# To use RandomizedSearchCV, need to create a parameter grid to sample from during fitting
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=50, stop=5000, num=20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 4, 8, 16]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid - at each iteration, a different combination of parameters will be chosen.
# Not every combination is chosen, but the random selection of combinations samples a wide range of values
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

hyper_rf_random = RandomizedSearchCV(estimator=randomforest, param_distributions=random_grid, n_iter=100, cv=10,
                                     verbose=2, random_state=42)
hyper_rf_random.fit(x, y)
hyper_1000of3200_best_params = hyper_rf_random.best_params_
hyper = DataFrame([hyper_1000of3200_best_params])
hyper.to_csv('data/hppy_rf_baseline_best_hypers_1000of3200.csv', index=False)

hyper_randomforest = RandomForestClassifier(n_estimators=2655, min_samples_split=4, min_samples_leaf=8, max_depth=90,
                                            random_state=42)

# hyperparameterised baseline random forest - training:test 4:1, full set of features, default hyperparameters
print('Hyper-ed baseline start')
hyper_baseline_stats, hyper_baseline_models = run_model_multiple(hyper_randomforest, x, y, test_size=0.2,
                                                                 num_iterations=100)
hyper_baseline_stats_df = DataFrame([hyper_baseline_stats])
print('Hyper-ed baseline complete')

hyper_baseline_stats_df.to_csv('data/hppy_hyper_baseline_rf.csv', index=False)

print('Hyper-ed Training90 start')
hyper_training90_stats, hyper_training90_models = run_model_multiple(hyper_randomforest, x, y, test_size=0.1,
                                                                     num_iterations=100)
hyper_training90_stats_df= DataFrame([hyper_training90_stats])
print('Hyper-ed Training90 complete')

hyper_training90_stats_df.to_csv('data/hppy_hyper_training90_rf.csv', index=False)
