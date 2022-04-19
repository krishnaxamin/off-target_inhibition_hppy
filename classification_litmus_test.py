from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pandas import read_csv, DataFrame, Series, concat
from sklearn.model_selection import train_test_split

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# train the random forest
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

# train the logistic regression
logistic = LogisticRegression(random_state=42, max_iter=1000)

lr_ranked_features = read_csv('data/hppy_lr_baseline_features_ranked.csv', index_col=0)
lr_features94 = lr_ranked_features.index.values.tolist()[:94]
x_lr_features94 = x[lr_features94]

x_lr_train, x_lr_test, y_lr_train, y_lr_test = train_test_split(x_lr_features94, y, test_size=0.2, random_state=883)
logistic.fit(x_lr_train, y_lr_train)

# litmus test on ATP, 2 non-hydrolysable ATP analogues, 1 known inhibitor, 1 assumed non-inhibitor and 7 inhibitors
# not in CHEMBL
atp_smiles = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N'
dmx_5804_smiles = 'COCCOC1=CC=C(C=C1)C2=CN(C3=C2C(=O)NC=N3)C4=CC=CC=C4'  # suggested inhibitor
gne_220_smiles = 'CC1=C2C(=NC(=N1)C3=CN(N=C3)C)C4=C(N2)N=CC(=C4)C5=CC=C(C=C5)N6CCN(CC6)C'  # suggested inhibitor
ng25_smiles = 'CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)OC4=C5C=CNC5=NC=C4)C(F)(F)F'  # suggested inhibitor
tak1_map4k2_1_smiles = 'CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)OC4=NC=NC5=C4C=C(N5)C)C(F)(F)F'  # suggested inhibitor
hipk1_in3_smiles = 'CN1CCC2=CC(=C(C=C2C1)NC3=NC=C(C(=N3)NC4=C(C=CC=C4F)C(F)(F)F)C(=O)N)OC'  # suggested inhibitor
hipk1_in7_smiles = 'CC1(C2=C(C=CC(=C2)NC3=NC=C(C(=N3)NC(CO)C4=CC=CC=C4)C5=NN=CO5)C(=O)O1)C'  # suggested inhibitor
hg6_64_1_smiles = 'CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)C3=CC(=C(C=C3)C)C=CC4=CN=C5C(=C4OC)C=CN5)C(F)(F)F'  # suggested inhibitor
amp_pnp_smiles = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(NP(=O)(O)O)O)O)O)N'  # suggested inhibitor
amp_pcp_smiles = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(CP(=O)(O)O)O)O)O)N'  # suggested inhibitor
diproleandomycin_smiles = 'CCC(=O)O[C@H]1[C@H](C)O[C@@H](O[C@@H]2[C@@H](C)C(=O)O[C@H](C)[C@H](C)[C@H](OC(=O)CC)[C@@H](C)C(=O)[C@]3(CO3)C[C@H](C)[C@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@H]2C)C[C@@H]1OC'
chembl1164265_smiles = 'CN1CCN(c2ccc(-c3cnc4c(c3)N(Cc3c(F)ccc(F)c3Cl)CCN4)cn2)CC1'  # known inhibitor
aminopyrrolopyrimidine = 'C1CNCCC1C2=NC=C(S2)C3=CC(=C(N=C3)N)OCC4=CC=NC=C4'  # bound inhibitor in 5J5T crystal structure

smiles_list = [atp_smiles, dmx_5804_smiles, gne_220_smiles, ng25_smiles,tak1_map4k2_1_smiles, hipk1_in7_smiles,
               hipk1_in3_smiles, hg6_64_1_smiles, amp_pcp_smiles, amp_pnp_smiles, diproleandomycin_smiles,
               chembl1164265_smiles]

smiles_smi = DataFrame({'canonical_smiles': smiles_list,
                        'name': ['ATP', 'DMX-5804', 'GNE-220', 'NG25', 'TAK1/MAP4K2-1', 'HIPK1-IN-7', 'HIPK1-IN-3',
                                  'HG6-64-1', 'AMP-PCP', 'AMP-PNP', 'DIPRO', 'CHEMBL1164265']})
smiles_smi.to_csv('padel/molecule.smi', sep='\t', index=False, header=False)

descriptor = read_csv('padel/descriptors_output.csv')
fingerprints = descriptor.drop(['Name'], axis=1)

randomforest.predict(fingerprints[rf_features43])  # RF with 43 features
logistic.predict(fingerprints[lr_features94])  # LR with 94 features
randomforest.predict(fingerprints)  # RF with 881 features
