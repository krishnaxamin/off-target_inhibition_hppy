from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from pandas import read_csv, DataFrame, concat
from scikit_regressor_funcs import run_regressor_multiple

df = read_csv('data/happyhour_inhibitor_name_activity_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'standard_value'], axis=1)

y = df['standard_value']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

# initialise random forest classifier
randomforest = RandomForestRegressor(random_state=42)

rf_feature_selector = RFECV(estimator=randomforest)
rf_feature_selector.fit(x, y)
rf_feature_rankings_array = rf_feature_selector.ranking_
rf_features_ranked = DataFrame(rf_feature_rankings_array, columns=['rank'])
rf_features_ranked = concat([rf_features_ranked, pubchem_fingerprints], axis=1).drop(['bit'], axis=1)
rf_features_ranked.index = x.columns.values.tolist()
rf_features_ranked = rf_features_ranked.sort_values(by='rank')
rf_features_ranked.to_csv('data/hppy_rf_regress_baseline_features_ranked.csv')

# get performance of RF with successively more and more features, starting from the most important
rf_features_ranked = read_csv('data/hppy_rf_regress_baseline_features_ranked.csv', index_col=0)
ordered_features = rf_features_ranked.index.values.tolist()
rf_performance_features_plot_list = []
for i in range(len(ordered_features)):
    num_features = i + 1
    features_to_use = ordered_features[:num_features]
    x_with_features_to_use = x[features_to_use]
    rf_stats, rf_models = run_regressor_multiple(randomforest, x_with_features_to_use, y,
                                                   test_size=0.2,
                                                   num_iterations=100,
                                                   model_type='Random Forest',
                                                   verbose=False)
    rf_r2 = rf_stats['mean_r2']
    rf_rmse = rf_stats['mean_rmse']
    rf_mae = rf_stats['mean_mae']
    rf_performance_features_plot_list.append([num_features, rf_r2, rf_rmse, rf_mae])
    print([num_features, rf_r2, rf_rmse, rf_mae])

rf_performance_features_plot_df = DataFrame(rf_performance_features_plot_list, columns=['num_features', 'mean_r2',
                                                                                        'mean_rmse',
                                                                                        'mean_mae'])
rf_performance_features_plot_df.to_csv('data/hppy_rf_regress_diff_num_features.csv', index=False)
