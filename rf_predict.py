from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split
from numpy import vstack
from scikit_classifier_funcs import data_balancer

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# UKB data
ukb_drugs_descriptor = read_csv('data/drug_ukb_name_fingerprints.csv')
ukb_drugs_fingerprints = ukb_drugs_descriptor.drop(['Name'], axis=1)
ukb_drugs_notna = read_csv('data/drug_ukb_cleaned.csv')

########################################################################################################################
# TRAIN

# train the random forest with 43 features
randomforest = RandomForestClassifier(random_state=42)

rf_ranked_features = read_csv('data/hppy_rf_baseline_features_ranked.csv', index_col=0)
rf_features43 = rf_ranked_features.index.values.tolist()[:43]
x_rf_features43 = x[rf_features43]

x_rf_train, x_rf_test, y_rf_train, y_rf_test = train_test_split(x_rf_features43, y, test_size=0.2, random_state=197)
randomforest.fit(x_rf_train, y_rf_train)

# train the random forest with the full set of features
randomforest = RandomForestClassifier(random_state=42)
x_rf_baseline_train, x_rf_baseline_test, y_rf_baseline_train, y_rf_baseline_test = train_test_split(x, y, test_size=0.2, random_state=197)
randomforest.fit(x_rf_baseline_train, y_rf_baseline_train)

# train the random forest with 35 features - balanced
randomforest = RandomForestClassifier(random_state=42)

rf_ranked_features = read_csv('data/hppy_rf_baseline_features_ranked.csv', index_col=0)
rf_features35 = rf_ranked_features.index.values.tolist()[:35]
x_rf_features35 = x[rf_features35]

x_rf_train, x_rf_test, y_rf_train, y_rf_test = train_test_split(x_rf_features35, y, test_size=0.2, random_state=981)
x_rf_train, y_rf_train = data_balancer(x_rf_train, y_rf_train)
randomforest.fit(x_rf_train, y_rf_train)

# train the random forest with the full set of features - balanced
randomforest = RandomForestClassifier(random_state=42)
x_rf_baseline_train, x_rf_baseline_test, y_rf_baseline_train, y_rf_baseline_test = train_test_split(x, y, test_size=0.2,
                                                                                                    random_state=559)
x_rf_baseline_train, y_rf_baseline_train = data_balancer(x_rf_baseline_train, y_rf_baseline_train)
randomforest.fit(x_rf_baseline_train, y_rf_baseline_train)

########################################################################################################################
# PREDICT

# predict which drugs inhibit hppy orthologues out of the UKB data
# RF imbalanced
# ukb_drugs_rf_predict = randomforest.predict(ukb_drugs_fingerprints[rf_features43])
ukb_drugs_rf_predict = randomforest.predict(ukb_drugs_fingerprints)
ukb_drugs_rf_classed = concat([ukb_drugs_notna.drop(['Drug', 'Drug_curated', 'smiles'], axis=1),
                               DataFrame(vstack(ukb_drugs_rf_predict), columns=['predicted_classification'])], axis=1)

rf_active_ukb_drugs = ukb_drugs_rf_classed[ukb_drugs_rf_classed['predicted_classification'] == 1]
rf_active_ukb_drugs.to_csv('data/hppy_rf_baseline_active_ukb_drugs.csv', index=False)

# RF balanced
ukb_drugs_rf_predict = randomforest.predict(ukb_drugs_fingerprints[rf_features35])
# ukb_drugs_rf_predict = randomforest.predict(ukb_drugs_fingerprints)
ukb_drugs_rf_classed = concat([ukb_drugs_notna.drop(['Drug', 'Drug_curated', 'smiles'], axis=1),
                               DataFrame(vstack(ukb_drugs_rf_predict), columns=['predicted_classification'])], axis=1)

rf_active_ukb_drugs = ukb_drugs_rf_classed[ukb_drugs_rf_classed['predicted_classification'] == 1]
rf_active_ukb_drugs.to_csv('data/hppy_rf_balanced981_35features_active_ukb_drugs.csv', index=False)
# rf_active_ukb_drugs.to_csv('data/hppy_rf_balanced559_baseline_active_ukb_drugs.csv', index=False)
