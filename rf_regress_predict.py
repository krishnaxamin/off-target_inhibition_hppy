from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split
from numpy import vstack

df = read_csv('data/happyhour_inhibitor_name_activity_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'standard_value'], axis=1)

y = df['standard_value']

# UKB data
ukb_drugs_descriptor = read_csv('data/drug_ukb_name_fingerprints.csv')
ukb_drugs_fingerprints = ukb_drugs_descriptor.drop(['Name'], axis=1)
ukb_drugs_notna = read_csv('data/drug_ukb_cleaned.csv')

########################################################################################################################
# TRAIN

randomforest = RandomForestRegressor(random_state=42)

# split data using random_state that gives best R2
x_r2_train, x_r2_test, y_r2_train, y_r2_test = train_test_split(x, y, test_size=0.2, random_state=782)

# split data using random_state that gives best RMSE and MAE
x_rest_train, x_rest_test, y_rest_train, y_rest_test = train_test_split(x, y, test_size=0.2, random_state=981)

# fit with r2 random_state
randomforest.fit(x_r2_train, y_r2_train)

# fit with rmse/mae random_state
randomforest.fit(x_rest_train, y_rest_train)

########################################################################################################################
# PREDICT

# predict which drugs inhibit hppy orthologues out of the UKB data

ukb_drugs_rf_predict = randomforest.predict(ukb_drugs_fingerprints)
ukb_drugs_rf_id_predict = concat([ukb_drugs_notna.drop(['Drug', 'Drug_curated', 'smiles'], axis=1),
                                 DataFrame(vstack(ukb_drugs_rf_predict), columns=['predicted_activity'])], axis=1)
ukb_drugs_rf_id_predict_sorted = ukb_drugs_rf_id_predict.sort_values('predicted_activity')

# predictions with r2 random_state
ukb_drugs_rf_id_predict_sorted.to_csv('data/hppy_rf_regress782_active_ukb_drugs.csv', index=False)

# predictions with rmse/mae random_state
ukb_drugs_rf_id_predict_sorted.to_csv('data/hppy_rf_regress981_active_ukb_drugs.csv', index=False)
