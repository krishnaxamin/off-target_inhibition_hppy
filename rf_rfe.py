from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from pandas import read_csv, DataFrame, concat
from scikit_classifier_funcs import run_model_multiple

"""
Script to perform recursive feature elimination with cross-validation (RFE-CV) for the random forest classifier.
RFE-CV ranks the features used in learning based on which has the most impact on an entry's predicted classification.
Then, we look at the performance of the classifier on successively more features, starting with the most important 
feature and adding the next most important until all the features are used.
"""

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

# initialise random forest classifier
randomforest = RandomForestClassifier(random_state=42)

rf_feature_selector = RFECV(estimator=randomforest)
rf_feature_selector.fit(x, y)
rf_feature_rankings_array = rf_feature_selector.ranking_
rf_features_ranked = DataFrame(rf_feature_rankings_array, columns=['rank'])
rf_features_ranked = concat([rf_features_ranked, pubchem_fingerprints], axis=1).drop(['bit'], axis=1)
rf_features_ranked.index = x.columns.values.tolist()
rf_features_ranked = rf_features_ranked.sort_values(by='rank')
rf_features_ranked.to_csv('data/hppy_rf_baseline_features_ranked.csv')

# get performance of RF with successively more and more features, starting from the most important
rf_features_ranked = read_csv('data/hppy_rf_baseline_features_ranked.csv', index_col=0)
ordered_features = rf_features_ranked.index.values.tolist()
rf_performance_features_plot_list = []
for i in range(len(ordered_features)):
    num_features = i + 1
    features_to_use = ordered_features[:num_features]
    x_with_features_to_use = x[features_to_use]
    rf_stats, rf_features, rf_ci95, rf_models = run_model_multiple(randomforest, x_with_features_to_use, y,
                                                                   test_size=0.2,
                                                                   num_iterations=100,
                                                                   model_type='Random Forest',
                                                                   verbose=False,
                                                                   balance_data=True)
    rf_accuracy = rf_stats['mean_accuracy']
    rf_sensitivity = rf_stats['mean_sensitivity']
    rf_specificity = rf_stats['mean_specificity']
    rf_balanced_accuracy = rf_stats['mean_balanced_accuracy']
    rf_f1 = rf_stats['mean_f1']
    rf_performance_features_plot_list.append([num_features, rf_accuracy, rf_sensitivity, rf_specificity,
                                              rf_balanced_accuracy, rf_f1])
    print([num_features, rf_accuracy, rf_sensitivity, rf_specificity, rf_balanced_accuracy, rf_f1])

rf_performance_features_plot_df = DataFrame(rf_performance_features_plot_list, columns=['num_features', 'mean_accuracy',
                                                                                        'mean_sensitivity',
                                                                                        'mean_specificity',
                                                                                        'mean_balanced_accuracy',
                                                                                        'mean_f1'])
rf_performance_features_plot_df.to_csv('data/hppy_rf_balanced_diff_num_features.csv', index=False)
