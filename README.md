# off-target_inhibition_hppy
Code used in my thesis for Part III (MSci) in Systems Biology at the University of Cambridge. In the lab of L. Miguel Martins at the MRC Toxicology Unit.

Using a combination of machine learning and molecular docking to suggest which drugs commonly used in the UK may be having off-target effects on human orthologues of happyhour and therefore increasing the risk of alcohol use disorder in those taking the suggested drugs. 

## classifier_plotting
Scripts to plot a:
* Neural net
* Decision tree
## data
All data generated by the .py files below. 
## docking
Contains:
* Executable for AutoDock Vina
* Vina configuration files for fgid and flexible docking
* Vina command 
* ADFRsuite commands to prepare a receptor for docking
* ADFRsuite commands to prepare ligands for docking
* ADFRsuite commands to define a search box for docking
## hppy_inhibition_r
Contains plots used in the write-up and R scripts for:
* Conversion of mol2 file format to pdb file format
* Plotting of data and graph generation
## padel
What's needed for the conversion of SMILES into PubChem fingerprints, using PaDEL-Descriptor:
* PaDEL-Descriptor files
* Java executable
* Command to run PaDEL-Descriptor as needed
## pymol_vis
Commands to get the 5 views (Views A-E) used to generate the PyMOL images. 
## scrap_py_files
.py files that I made but were not used in the project. Maintained here for my storage want more than anything else.
## source_data
Soure data files used by the .py files below.
## .py files
Scripts and a Colab notebook that carry out data preparation and machine learning capabilities and a script for coloring alignments in PyMOL by distance between alpha carbon atoms.
### colorbyrmsd.py
'Align two structures and show the structural deviations in color to more easily see variable regions. Colors each mobile/target atom-pair by distance (the name is a bit misleading). Modifies the B-factor columns in your original structures.'
[Source](http://pymolwiki.org/index.php/ColorByRMSD)
### data_prep.py
Script to collect and clean data from ChEMBL and UKB. Data to be used for training and testing, and to make predictions on.
### dnn_class.py
Script to run tuned models, get their losses on training and validation data over their training, their performance on test data and their predictions on query UKB data.
### dnn_class_colab.ipynb
Colab notebook that repliates dnn_class.py. Tuned models had to be run on Colab as my laptop did not have the memory for it.
### dnn_tuning.py
Script to use keras-tuner for hyperparameterisation. Train-val-test splits can be varied.
Things tuned:
* number of dense hidden layers (between 1 and 3)
* number of neurons in each dense hidden layer (between 32 and 512, in increments of 32)
* the activation function of neurons in the dense hidden layers (relu or tanh)
* the presence of a Dropout layer after the hidden layers
* the dropout rate (0.25, 0.5, 0.75)
* the learning rate (between 1e-10 and 1, with log sampling)
### lr_predict.py
Uses a selection of LR models to make predictions on UKB data and all approved small molecule drugs from ChEMBL.
### lr_rfe.py
Script to perform recursive feature elimination with cross-validation (RFE-CV) for the logistic regression classifier. RFE-CV ranks the features used in learning based on which has the most impact on an entry's predicted classification. It then looks at performance of the classifier using different sets of ranked features, e.g. the top 10 features. Finally, it looks at the performance of the classifier on successively more features, starting with the most important feature and adding the next most important until all the features are used.
### plot_trees.py
Script to make .dot files for graphviz plotting of decision trees. Decision trees plotted are the first ones in the forests, which are trained on training data generated using seeds 678 and 981. .dot files are then converted to visualisation files in the terminal.
### rf_predict.py 
Uses a selection of RF models to make predictions on UKB data.
### rf_rfe.py
Script to perform recursive feature elimination with cross-validation (RFE-CV) for the random forest classifier. RFE-CV ranks the features used in learning based on which has the most impact on an entry's predicted classification. Then, we look at the performance of the classifier on successively more features, starting with the most important feature and adding the next most important until all the features are used.
### scikit_classifier_exploration.py
Script to run random forest and logistic regression classifiers of varying types. 
### scikit_classifier_funcs.py
Collection of functions used wherever scikit-learn classifiers are involved elsewhere in the project.
### scikit_diff_classifiers.py
Script to get the performance of different classifiers available in scikit-learn. The classifiers are used in their default setting. We also look at the average time taken for a single classifier to train, averaged over 100 runs of that classifier, although this is not used further.
