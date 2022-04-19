from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pandas import read_csv, DataFrame
from scikit_classifier_funcs import run_model_multiple

"""
Script to run random forest and logistic regression classifiers of varying types. 
"""

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

########################################################################################################################
# LOGISTIC REGRESSION

# initialise logistic regression classifier
logistic = LogisticRegression(random_state=42, max_iter=1000)

# baseline LR - train-test split of 80-20, 881 features, default hyperparameters
print('Baseline LR start')
baseline_lr_stats, baseline_lr_models = run_model_multiple(logistic, x, y,
                                                           test_size=0.2,
                                                           num_iterations=100,
                                                           model_type='Logistic Regression',
                                                           balance_data=True)
baseline_lr_df = DataFrame([baseline_lr_stats])
print('Baseline LR complete')
baseline_lr_df.to_csv('data/hppy_lr_balanced_baseline.csv', index=False)
baseline_lr_models.to_csv('data/hppy_lr_balanced_baseline_databymodel.csv', index=False)

# LR with train-test split of 90-10
print('Training90 LR start')
training90_lr_stats, training90_lr_models = run_model_multiple(logistic, x, y,
                                                               test_size=0.1,
                                                               num_iterations=100,
                                                               model_type='Logistic Regression',
                                                               balance_data=True)
training90_lr_stats_df = DataFrame([training90_lr_stats])
print('Training90 LR complete')
training90_lr_stats_df.to_csv('data/hppy_lr_balanced_training90.csv', index=False)
training90_lr_models.to_csv('data/hppy_lr_balanced_training90_databymodel.csv', index=False)

# LR with train-test split of 70-30
print('Training70 LR start')
training70_lr_stats, training70_lr_models = run_model_multiple(logistic, x, y,
                                                               test_size=0.3,
                                                               num_iterations=100,
                                                               model_type='Logistic Regression',
                                                               balance_data=True)
training70_lr_stats_df = DataFrame([training70_lr_stats])
print('Training70 LR complete')
training70_lr_stats_df.to_csv('data/hppy_lr_balanced_training70.csv', index=False)

# num_features = 94 - using imbalanced data
lr_ranked_features = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
lr_features94 = lr_ranked_features.index.values.tolist()[:94]
x_lr_features94 = x[lr_features94]

print('Best LR start')
best_lr_stats, best_lr_models = run_model_multiple(logistic, x_lr_features94, y,
                                                   test_size=0.2,
                                                   num_iterations=100,
                                                   model_type='Logistic Regression',
                                                   balance_data=True)
best_lr_stats_df = DataFrame([best_lr_stats])
print('Best LR complete')
best_lr_stats_df.to_csv('data/hppy_lr_best.csv', index=False)

# sort the 100 runs of this LR model by accuracy, to find the random_state that gives best model performance (in terms
# of all 5 metrics used to evaluate performance)
best_lr_models_ranked = best_lr_models.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
best_lr_models_ranked.to_csv('data/hppy_lr_94features_rankedmodels.csv', index=False)

# num_features = 98 - using balanced data
lr_ranked_features = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
lr_features98 = lr_ranked_features.index.values.tolist()[:98]
x_lr_features98 = x[lr_features98]

print('Best LR start')
best_lr_stats, best_lr_models = run_model_multiple(logistic, x_lr_features98, y,
                                                   test_size=0.2,
                                                   num_iterations=100,
                                                   model_type='Logistic Regression',
                                                   balance_data=True,
                                                   track_features=True)
best_lr_stats_df = DataFrame([best_lr_stats])
print('Best LR complete')
best_lr_stats_df.to_csv('data/hppy_lr_balanced_best.csv', index=False)

# save the run-by-run data
best_lr_models.to_csv('data/hppy_lr_balanced_best_databymodel.csv', index=False)

# as above
# find the random_state that gives best model performance (in terms of all 5 metrics used to evaluate performance)
best_lr_models_ranked = best_lr_models.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
best_lr_models_ranked.to_csv('data/hppy_lr_balanced_98features_rankedmodels.csv', index=False)

########################################################################################################################
# RANDOM FOREST

# initialise random forest classifier
randomforest = RandomForestClassifier(random_state=42)

# baseline random forest - train-test split of 80-20, 881 features, default hyperparameters
print('Baseline start')
baseline_stats, baseline_models = run_model_multiple(randomforest, x, y,
                                                     test_size=0.2,
                                                     num_iterations=100,
                                                     balance_data=True,
                                                     track_features=True)
baseline_stats_df = DataFrame([baseline_stats])
print('Baseline complete')
baseline_stats_df.to_csv('data/hppy_rf_balanced_baseline.csv', index=False)
baseline_models.to_csv('data/hppy_rf_balanced_baseline_databymodel.csv', index=False)

# train-test split of 90-10
print('Training90 start')
training90_stats, training90_models = run_model_multiple(randomforest, x, y,
                                                         test_size=0.1,
                                                         num_iterations=100,
                                                         balance_data=True)
training90_stats_df = DataFrame([training90_stats])
print('Training90 complete')

training90_stats_df.to_csv('data/hppy_rf_balanced_training90.csv', index=False)
training90_models.to_csv('data/hppy_rf_balanced_training90_databymodel.csv', index=False)

# num_features = 43 - imbalanced data
rf_ranked_features = read_csv('data/hppy_rf_baseline_features_ranked.csv', index_col=0)
rf_features43 = rf_ranked_features.index.values.tolist()[:43]
x_rf_features43 = x[rf_features43]

print('Best RF start')
best_rf_stats, best_rf_models = run_model_multiple(randomforest, x_rf_features43, y, test_size=0.2,
                                                   num_iterations=100, model_type='Random Forest')
best_rf_stats_df = DataFrame([best_rf_stats])
print('Best RF complete')

# as above
# find the random_state that gives best model performance (in terms of all 5 metrics used to evaluate performance)
best_rf_models_ranked = best_rf_models.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
best_rf_models_ranked.to_csv('data/hppy_rf_baseline_43features_rankedmodels.csv', index=False)

# num_features = 35 - balanced data
rf_ranked_features = read_csv('data/hppy_rf_baseline_features_ranked.csv', index_col=0)
rf_features35 = rf_ranked_features.index.values.tolist()[:35]
x_rf_features35 = x[rf_features35]

print('Best RF start')
best_rf_stats, best_rf_models = run_model_multiple(randomforest, x_rf_features35, y,
                                                   test_size=0.2,
                                                   num_iterations=100,
                                                   model_type='Random Forest',
                                                   balance_data=True,
                                                   track_features=True)
best_rf_stats_df = DataFrame([best_rf_stats])
print('Best RF complete')

best_rf_stats_df.to_csv('data/hppy_rf_balanced_best.csv', index=False)
best_rf_models.to_csv('data/hppy_rf_balanced_best_databymodel.csv', index=False)

# as above
# find the random_state that gives best model performance (in terms of all 5 metrics used to evaluate performance)
best_rf_models_ranked = best_rf_models.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
best_rf_models_ranked.to_csv('data/hppy_rf_balanced_35features_rankedmodels.csv', index=False)

# best RF model
# num_features = 881 - imbalanced data
randomforest = RandomForestClassifier(random_state=42)

print('Best RF start')
best_rf_baseline_stats, best_rf_baseline_models = run_model_multiple(randomforest, x, y, test_size=0.2,
                                                                     num_iterations=100, model_type='Random Forest')
best_rf_baseline_stats_df = DataFrame([best_rf_baseline_stats])
print('Best RF complete')

# as above
# find the random_state that gives best model performance (in terms of all 5 metrics used to evaluate performance)
best_rf_baseline_models_ranked = best_rf_baseline_models.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
best_rf_baseline_models_ranked.to_csv('data/hppy_rf_baseline_rankedmodels.csv', index=False)

# num_features = 881 - balanced data
randomforest = RandomForestClassifier(random_state=42)

print('Best RF start')
best_rf_baseline_stats, best_rf_baseline_models = run_model_multiple(randomforest, x, y, test_size=0.2,
                                                                     num_iterations=100, model_type='Random Forest',
                                                                     balance_data=True)
best_rf_baseline_stats_df = DataFrame([best_rf_baseline_stats])
print('Best RF complete')

# as above
# find the random_state that gives best model performance (in terms of all 5 metrics used to evaluate performance)
best_rf_baseline_models_ranked = best_rf_baseline_models.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
best_rf_baseline_models_ranked.to_csv('data/hppy_rf_balanced_baseline_rankedmodels.csv', index=False)
