from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from pandas import read_csv, DataFrame, concat
from scikit_classifier_funcs import run_model_multiple

"""
Script to perform recursive feature elimination with cross-validation (RFE-CV) for the logistic regression classifier.
RFE-CV ranks the features used in learning based on which has the most impact on an entry's predicted classification.
It then looks at performance of the classifier using different sets of ranked features, e.g. the top 10 features. 
Finally, it looks at the performance of the classifier on successively more features, starting with the most important 
feature and adding the next most important until all the features are used.
"""

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')


logistic = LogisticRegression(random_state=42, max_iter=1000)

# ranking the features in the logistic regression
lr_feature_selector = RFECV(estimator=logistic)
lr_feature_selector.fit(x, y)
lr_feature_rankings_array = lr_feature_selector.ranking_
lr_feature_rankings_df = DataFrame([lr_feature_rankings_array], columns=x.columns.values.tolist())
lr_feature_rankings_df.to_csv('data/hppy_lr_baseline_feature_ranks.csv', index=False)
lr_features_ranked = DataFrame(lr_feature_rankings_array, columns=['rank'])
lr_features_ranked = concat([lr_features_ranked, pubchem_fingerprints], axis=1).drop(['bit'], axis=1)
lr_features_ranked.index = x.columns.values.tolist()
lr_features_ranked = lr_features_ranked.sort_values(by='rank')
lr_features_ranked.to_csv('data/hppy_lr_baseline_features_ranked.csv')

# getting the top 10 features while eliminating features 1-by-1
lr_10_features_selector = RFECV(estimator=logistic, min_features_to_select=10)
lr_10_features_selector.fit(x, y)
lr_10_features_feature_rankings_array = lr_10_features_selector.ranking_
lr_10_features_features_ranked = DataFrame(lr_10_features_feature_rankings_array, columns=['rank'])
lr_10_features_features_ranked = concat([lr_10_features_features_ranked, pubchem_fingerprints], axis=1).drop(['bit'], axis=1)
lr_10_features_features_ranked.index = x.columns.values.tolist()
lr_10_features_features_ranked = lr_10_features_features_ranked.sort_values(by='rank')
lr_10_features_features_ranked.to_csv('data/hppy_lr_baseline_10features_ranked.csv', index=False)

# getting the top 10 features while eliminating 25% of features remaining at a time
lr_10_features_remove25percent_selector = RFECV(estimator=logistic, step=0.25, min_features_to_select=10)
lr_10_features_remove25percent_selector.fit(x, y)
lr_10_features_remove25percent_feature_rankings_array = lr_10_features_remove25percent_selector.ranking_
lr_10_features_remove25percent_features_ranked = DataFrame(lr_10_features_remove25percent_feature_rankings_array,
                                                           columns=['rank'])
lr_10_features_remove25percent_features_ranked = concat([lr_10_features_remove25percent_features_ranked,
                                                         pubchem_fingerprints], axis=1).drop(['bit'], axis=1)
lr_10_features_remove25percent_features_ranked.index = x.columns.values.tolist()
lr_10_features_remove25percent_features_ranked = lr_10_features_remove25percent_features_ranked.sort_values(by='rank')
lr_10_features_remove25percent_features_ranked.to_csv('data/hppy_lr_baseline_10features_remove25percent_ranked.csv',
                                                      index=False)

# get performance of LR with successively more and more features, starting from the most important
lr_features_ranked = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
ordered_features = lr_features_ranked.index.values.tolist()
lr_performance_features_plot_list = []
for i in range(len(ordered_features)):
    num_features = i + 1
    features_to_use = ordered_features[:num_features]
    x_with_features_to_use = x[features_to_use]
    lr_stats, lr_models = run_model_multiple(logistic, x_with_features_to_use, y, test_size=0.2, num_iterations=100,
                                             model_type='Logistic Regression', verbose=False, balance_data=True)
    lr_accuracy = lr_stats['mean_accuracy']
    lr_sensitivity = lr_stats['mean_sensitivity']
    lr_specificity = lr_stats['mean_specificity']
    lr_balanced_accuracy = lr_stats['mean_balanced_accuracy']
    lr_f1 = lr_stats['mean_f1']
    lr_performance_features_plot_list.append([num_features, lr_accuracy, lr_sensitivity, lr_specificity,
                                              lr_balanced_accuracy, lr_f1])
    print([num_features, lr_accuracy, lr_sensitivity, lr_specificity, lr_balanced_accuracy, lr_f1])

lr_performance_features_plot_df = DataFrame(lr_performance_features_plot_list, columns=['num_features', 'mean_accuracy',
                                                                                        'mean_sensitivity',
                                                                                        'mean_specificity',
                                                                                        'mean_balanced_accuracy',
                                                                                        'mean_f1'])
lr_performance_features_plot_df.to_csv('data/hppy_lr_balanced_diff_num_features2.csv', index=False)
