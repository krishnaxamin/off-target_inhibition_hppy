from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from pandas import read_csv, DataFrame
from time import time
from scikit_classifier_funcs import run_model_multiple

"""
Script to get the performance of different classifiers available in scikit-learn. The classifiers are used in their 
default setting. We also look at the average time taken for a single classifier to train, averaged over 100 runs of that
classifier, although this is not used further.
"""


df = read_csv('data/happyhour_inhibitor_name_class_fingerprints.csv')

# input data - features
x = df.drop(['molecule_chembl_id', 'classification'], axis=1)

# binary output data for classification - class (active, inactive)
y = df['classification']

# get fingerprints
pubchem_fingerprints = read_csv('data/pubchem_fingerprints.csv')

print('Different models: begin')
randomforest = RandomForestClassifier(random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)
logistic = LogisticRegression(random_state=42, max_iter=1000)
kneighbours = KNeighborsClassifier()
naive_bayes = GaussianNB()
svc = SVC(random_state=42)
bernoulli = BernoulliNB()
adaboost = AdaBoostClassifier(random_state=42)
gradboosting = GradientBoostingClassifier(random_state=42)
histgradboosting = HistGradientBoostingClassifier(random_state=42)
quadratic = QuadraticDiscriminantAnalysis()
mlp = MLPClassifier(random_state=42)
gaussian = GaussianProcessClassifier(random_state=42)

model_names = ['Random Forest',
               'Decision Tree',
               'Logistic Regression',
               'K Nearest Neighbours',
               'Naive Bayes',
               'SVC',
               'BernoulliNB',
               'AdaBoost',
               'Gradient Boosting',
               'Hist. Gradient Boosting',
               'QDA',
               'Neural Net - MLP',
               'GPC']

model_list = [randomforest, decision_tree, logistic, kneighbours, naive_bayes, svc, bernoulli, adaboost,
              gradboosting, histgradboosting, quadratic, mlp, gaussian]

class_models_df = DataFrame()
for idx, model in enumerate(model_list):
    print(model_names[idx])
    unbal_model_stats, unbal_model_models = run_model_multiple(model, x, y, num_iterations=100,
                                                               model_type=model_names[idx], balance_data=False)
    bal_model_stats, bal_model_models = run_model_multiple(model, x, y, num_iterations=100,
                                                           model_type=model_names[idx], balance_data=True)
    unbal_model_models.to_csv('data/hppy_' + model_names[idx] + '_unbalanced.csv', index=False)
    bal_model_models.to_csv('data/hppy_' + model_names[idx] + '_balanced.csv', index=False)
    # model_stats_df = DataFrame([model_stats])
    # class_models_df = concat([class_models_df, model_stats_df])
print('Different models: complete')

# class_models_df.to_csv('data/hppy_different_models_balanced.csv', index=False)

# get times taken for model training

histgradboost_start = time()
run_model_multiple(histgradboosting, x, y, num_iterations=100, model_type='Hist. Gradient Boosting', balance_data=True)
histgradboost_end = time()
histgradboost_time_taken = histgradboost_end - histgradboost_start

gradboost_start = time()
run_model_multiple(gradboosting, x, y, num_iterations=100, model_type='Gradient Boosting', balance_data=True)
gradboost_end = time()
gradboost_time_taken = gradboost_end - gradboost_start

rf_start = time()
run_model_multiple(randomforest, x, y, num_iterations=100, model_type='Random Forest', balance_data=True)
rf_end = time()
rf_time_taken = rf_end - rf_start

nn_start = time()
run_model_multiple(mlp, x, y, num_iterations=100, model_type='Neural Net - MLP', balance_data=True)
nn_end = time()
nn_time_taken = nn_end - nn_start

log_start = time()
run_model_multiple(logistic, x, y, num_iterations=100, model_type='Logistic Regression', balance_data=True)
log_end = time()
log_time_taken = log_end - log_start
log_time_taken/100

histgradboost_time_taken
gradboost_time_taken
rf_time_taken
nn_time_taken

time_taken_models = DataFrame({'model': ['Hist. Gradient Boosting', 'Gradient Boosting', 'Random Forest',
                                         'Neural Net - MLP'],
                              'time_taken': [histgradboost_time_taken/100, gradboost_time_taken/100, rf_time_taken/100,
                                             nn_time_taken/100]})
time_taken_models.to_csv('data/hppy_different_models_time_taken.csv', index=False)
