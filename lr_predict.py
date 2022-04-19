from sklearn.linear_model import LogisticRegression
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split
from numpy import vstack
from scikit_classifier_funcs import data_balancer

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# all small molecules that have been approved for use, as per ChEMBL
approved_drugs_small_mols_fingerprints = read_csv('data/active_small_molecule_drugs_name_fingerprints.csv')
approved_drugs_small_mols = approved_drugs_small_mols_fingerprints.drop(['Name'], axis=1)

# UKB data
ukb_drugs_descriptor = read_csv('data/drug_ukb_name_fingerprints.csv')
ukb_drugs_fingerprints = ukb_drugs_descriptor.drop(['Name'], axis=1)
ukb_drugs_notna = read_csv('data/drug_ukb_cleaned.csv')

########################################################################################################################
# TRAIN

# train the logistic regression - imbalanced
logistic = LogisticRegression(random_state=42, max_iter=1000)

lr_ranked_features = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
lr_features94 = lr_ranked_features.index.values.tolist()[:94]
x_lr_features94 = x[lr_features94]

x_lr_train, x_lr_test, y_lr_train, y_lr_test = train_test_split(x_lr_features94, y, test_size=0.2, random_state=883)
logistic.fit(x_lr_train, y_lr_train)

# train the logistic regression - balanced
logistic = LogisticRegression(random_state=42, max_iter=1000)

lr_ranked_features = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
lr_features98 = lr_ranked_features.index.values.tolist()[:98]
x_lr_features98 = x[lr_features98]

x_lr_train, x_lr_test, y_lr_train, y_lr_test = train_test_split(x_lr_features98, y, test_size=0.2, random_state=772)
x_lr_train, y_lr_train = data_balancer(x_lr_train, y_lr_train)
logistic.fit(x_lr_train, y_lr_train)

########################################################################################################################
# PREDICT

# predict which of all approved small molecules inhibit hppy orthologues
approved_drugs_small_mols_lr_predict = logistic.predict(approved_drugs_small_mols_fingerprints[lr_features94])
approved_drugs_small_mols_lr_classed = concat([approved_drugs_small_mols,
                                               DataFrame(vstack(approved_drugs_small_mols_lr_predict), columns=['predicted_classification'])], axis=1)

lr_active_approved_drugs_small_mols = approved_drugs_small_mols_lr_classed[approved_drugs_small_mols_lr_classed['predicted_classification'] == 1]
lr_active_approved_drugs_small_mols.to_csv('data/hppy_lr_active_small_molecule_drugs.csv', index=False)

# predict which drugs inhibit hppy orthologues out of the UKB data
# LR imbalanced
ukb_drugs_lr_predict = logistic.predict(ukb_drugs_fingerprints[lr_features94])
ukb_drugs_lr_classed = concat([ukb_drugs_notna.drop(['Drug', 'Drug_curated', 'smiles'], axis=1),
                               DataFrame(vstack(ukb_drugs_lr_predict), columns=['predicted_classification'])], axis=1)

lr_active_ukb_drugs = ukb_drugs_lr_classed[ukb_drugs_lr_classed['predicted_classification'] == 1]
lr_active_ukb_drugs.to_csv('data/hppy_lr_baseline_active_ukb_drugsv2.csv', index=False)

# LR balanced
ukb_drugs_lr_predict = logistic.predict(ukb_drugs_fingerprints[lr_features98])
ukb_drugs_lr_classed = concat([ukb_drugs_notna.drop(['Drug', 'Drug_curated', 'smiles'], axis=1),
                               DataFrame(vstack(ukb_drugs_lr_predict), columns=['predicted_classification'])], axis=1)

lr_active_ukb_drugs = ukb_drugs_lr_classed[ukb_drugs_lr_classed['predicted_classification'] == 1]
lr_active_ukb_drugs.to_csv('data/hppy_lr_balanced722_active_ukb_drugs.csv', index=False)
