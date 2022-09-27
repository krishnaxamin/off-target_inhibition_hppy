from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from pandas import read_csv, DataFrame, concat
from scikit_regressor_funcs import run_regressor_multiple

######
# IC50

ic50 = read_csv('data/happyhour_inhibitor_name_ic50_fingerprints.csv')

# input data - features
x_ic50 = ic50.drop(['molecule_chembl_id', 'standard_value'], axis=1)

# output values
y_ic50 = ic50['standard_value']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

# split data into 80-20 train-test
x_ic50_train_80, x_ic50_test_20, y_ic50_train_80, y_ic50_test_20 = train_test_split(x_ic50, y_ic50, test_size=0.2)

# RF regressor
randomforest = RandomForestRegressor(random_state=42)

rf_ic50_baseline_stats, rf_ic50_baseline_features = run_regressor_multiple(randomforest, x_ic50, y_ic50,
                                                                           num_iterations=100,
                                                                           track_individual_models=False,
                                                                           track_features=True)
rf_ic50_baseline_stats_df = DataFrame([rf_ic50_baseline_stats])
rf_ic50_baseline_stats_df.to_csv('data/hppy_rf_regress_ic50_baseline.csv', index=False)

######
# activity

activity = read_csv('data/happyhour_inhibitor_name_activity_fingerprints.csv')

# input data - features
x_activity = activity.drop(['molecule_chembl_id', 'standard_value'], axis=1)

# output values
y_activity = activity['standard_value']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

# split data into 80-20 train-test
x_activity_train_80, x_activity_test_20, y_activity_train_80, y_activity_test_20 = train_test_split(x_activity, y_activity, test_size=0.2)

# RF regressor
randomforest = RandomForestRegressor(random_state=42)

rf_activity_baseline_stats, rf_activity_baseline_models = run_regressor_multiple(randomforest, x_activity, y_activity,
                                                                                   num_iterations=100,
                                                                                   track_individual_models=True,
                                                                                   track_features=True)
rf_activity_baseline_stats_df = DataFrame([rf_activity_baseline_stats])
rf_activity_baseline_stats_df.to_csv('data/hppy_rf_regress_activity_baseline.csv', index=False)
rf_activity_baseline_models.to_csv('data/hppy_rf_regress_activity_baseline_databymodel.csv', index=False)
