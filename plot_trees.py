from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pandas import read_csv, concat
from math import floor
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from subprocess import call

""" 
Script to make .dot files for graphviz plotting of decision trees. Decision trees plotted are the first ones in 
the forests, which are trained on training data generated using seeds 678 and 981. .dot files are then converted to 
visualisation files in the terminal.
"""

df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']


# oversample minority class (inhibitors) in imbalanced data to give a more balanced dataset
# input_x is a df, input_y is a numpy array
def data_balancer(input_x, input_y):
    num_inhibitors = sum(input_y)
    num_non_inhibitors = len(input_y) - num_inhibitors
    if num_non_inhibitors/num_inhibitors < 2:  # data sufficiently balanced
        x_out = input_x
        y_out = input_y
    else:   # data imbalanced
        # x = DataFrame(input_x)
        y = input_y.to_frame()
        data = concat([input_x, y], axis=1)
        times_to_replicate = floor(num_non_inhibitors/num_inhibitors) - 1
        inhibitors = data[data['classification'] == 1]
        inhibitors_replicated = concat([inhibitors]*times_to_replicate, ignore_index=True)
        data_balanced = concat([data, inhibitors_replicated], ignore_index=True)
        x_out = data_balanced.drop(['classification'], axis=1)
        y_out = data_balanced['classification']
    return x_out, y_out


rf_ranked_features = read_csv('data/hppy_rf_baseline_features_ranked.csv', index_col=0)
rf_features35 = rf_ranked_features.index.values.tolist()[:35]
x_rf_features35 = x[rf_features35]

randomforest = RandomForestClassifier(random_state=42)

x_rf_train, x_rf_test, y_rf_train, y_rf_test = train_test_split(x_rf_features35, y, test_size=0.2, random_state=678)
x_rf_train, y_rf_train = data_balancer(x_rf_train, y_rf_train)
randomforest.fit(x_rf_train, y_rf_train)

estimator_to_plot = randomforest.estimators_[0]

fn = x_rf_train.columns.tolist()
cn = ['non-inhibitory', 'inhibitory']

export_graphviz(estimator_to_plot,
                out_file='depth-3_678.dot',
                max_depth=3,
                feature_names=fn,
                class_names=cn,
                rounded=True,
                filled=True)


