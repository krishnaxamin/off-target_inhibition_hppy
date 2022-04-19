"
Script to collect data, do some analysis and plot graphs. 
"

# PCA of ChEMBL data ------
library(tidyverse)

data <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/happyhour_inhibitor_name_class_fingerprints.csv')

fingerprints <- data[, 3:883]

## PCA on the full dataset ------

pca <- prcomp(fingerprints, center=TRUE)
pca_eigenvectors <- pca$rotation
pca_eigenvalues <- pca$sdev ^ 2
pca_percentages <- pca_eigenvalues/sum(pca_eigenvalues)

pca_points <- pca$x

ggplot(data.frame(pca_points), aes(PC1, PC2)) +
  geom_point(aes(color=factor(data$classification))) +
  labs(colour='Classification') +
  scale_color_discrete(labels=c('Non-inhibitory', 'Inhibitory'))

## PCA when non-variable features are removed (553 features left) ------

sums <- colSums(fingerprints)
zero_fingers <- which(sums == 0)
full_fingers <- which(sums == 909)
fingerprints_cut <- fingerprints %>%
  select(-names(zero_fingers)) %>%
  select(-names(full_fingers))

pca_cut <- prcomp(fingerprints_cut, center=TRUE)
pca_cut_eigenvectors <- pca_cut$rotation
pca_cut_eigenvalues <- pca_cut$sdev ^ 2
pca_cut_percentages <- pca_cut_eigenvalues/sum(pca_cut_eigenvalues)

pca_cut_points <- pca_cut$x

ggplot(data.frame(pca_cut_points), aes(PC1, PC2)) +
  geom_point(aes(color=factor(data$classification))) +
  labs(colour='Classification') +
  scale_color_discrete(labels=c('Non-inhibitory', 'Inhibitory'))

## PCA on LR's 98 features ------

lr_features <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_baseline_features_ranked.csv')
feature_names <- lr_features[1:98, ]$X
fingerprints_lr <- select(fingerprints, feature_names)

pca_lr <- prcomp(fingerprints_lr, center=TRUE)
pca_lr_eigenvectors <- pca_lr$rotation
pca_lr_eigenvalues <- pca_lr$sdev ^ 2
pca_lr_percentages <- pca_lr_eigenvalues/sum(pca_lr_eigenvalues)

pca_lr_points <- pca_lr$x

ggplot(data.frame(pca_lr_points), aes(PC1, PC2)) +
  geom_point(aes(color=factor(data$classification))) +
  labs(colour='Classification') +
  scale_color_discrete(labels=c('Non-inhibitory', 'Inhibitory'))

## PCA on RF's 35 features ------

rf_features <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_rf_baseline_features_ranked.csv')
feature_names <- rf_features[1:35, ]$X
fingerprints_rf <- select(fingerprints, feature_names)

pca_rf <- prcomp(fingerprints_rf, center=TRUE)
pca_rf_eigenvectors <- pca_rf$rotation
pca_rf_eigenvalues <- pca_rf$sdev ^ 2
pca_rf_percentages <- pca_rf_eigenvalues/sum(pca_rf_eigenvalues)

pca_rf_points <- pca_rf$x

ggplot(data.frame(pca_rf_points), aes(PC1, PC2)) +
  geom_point(aes(color=factor(data$classification))) +
  labs(colour='Classification') +
  scale_color_discrete(labels=c('Non-inhibitory', 'Inhibitory')) 

# Model performance with diff. num_features ------
library(tidyverse)

## LR ------

lr <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_balanced_diff_num_features.csv')
lr_long <- pivot_longer(lr, cols=!num_features, names_to='metric', values_to='value')
lr_long$metric <- str_replace_all(lr_long$metric, 
                                  c('mean_accuracy'='Mean Accuracy',
                                    'mean_balanced_accuracy'='Mean Balanced Accuracy',
                                    'mean_sensitivity'='Mean Sensitivity',
                                    'mean_specificity'='Mean Specificity',
                                    'mean_f1'='Mean F1 Score'))

# get the point at which each of the metrics peaks, for plotting later
lr_max_accuracy <- lr_long[lr_long$metric == 'Mean Accuracy' & lr_long$value == max(lr$mean_accuracy), ]
lr_max_balanced_accuracy <- lr_long[lr_long$metric == 'Mean Balanced Accuracy' & lr_long$value == max(lr$mean_balanced_accuracy), ]
lr_max_sensitivity <- lr_long[lr_long$metric == 'Mean Sensitivity' & lr_long$value == max(lr$mean_sensitivity), ]
lr_max_specificity <- lr_long[lr_long$metric == 'Mean Specificity' & lr_long$value == max(lr$mean_specificity), ]
lr_max_f1 <- lr_long[lr_long$metric == 'Mean F1 Score' & lr_long$value == max(lr$mean_f1), ]


ggplot(lr_long, aes(x=num_features, y=value, color=metric)) +
  geom_line() + 
  scale_color_brewer(palette='Dark2') +  # Dark2 used as it's colour-blind friendly
  labs(title='Impact of feature number of logistic regression performance',
       x='Number of features',
       y='Value',
       color='Metric') +
  geom_point(data = lr_max_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,  # shape=7 gives the box with a cross in it
             size = 3) +
  geom_point(data = lr_max_balanced_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_sensitivity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_specificity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_f1,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3)

# panel generation
ggplot(lr_long, aes(x=num_features, y=value, color=metric)) +
  geom_line() + 
  scale_color_brewer(palette='Dark2') +
  labs(title='A',  # I think title looks better as sub-panel labelling than tag
       x='Number of features',
       y='Value',
       color='Metric') +
  geom_point(data = lr_max_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_balanced_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_sensitivity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_specificity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = lr_max_f1,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) + 
  theme(legend.position = 'none')

## RF -------

rf <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_rf_balanced_diff_num_features.csv')
rf_long <- pivot_longer(rf, cols=!num_features, names_to='metric', values_to='value')
rf_long$metric <- str_replace_all(rf_long$metric, 
                                  c('mean_accuracy'='Mean Accuracy',
                                    'mean_balanced_accuracy'='Mean Balanced Accuracy',
                                    'mean_sensitivity'='Mean Sensitivity',
                                    'mean_specificity'='Mean Specificity',
                                    'mean_f1'='Mean F1 Score'))

rf_max_accuracy <- rf_long[rf_long$metric == 'Mean Accuracy' & rf_long$value == max(rf$mean_accuracy), ]
rf_max_balanced_accuracy <- rf_long[lr_long$metric == 'Mean Balanced Accuracy' & rf_long$value == max(rf$mean_balanced_accuracy), ]
rf_max_sensitivity <- rf_long[rf_long$metric == 'Mean Sensitivity' & rf_long$value == max(rf$mean_sensitivity), ]
rf_max_specificity <- rf_long[rf_long$metric == 'Mean Specificity' & rf_long$value == max(rf$mean_specificity), ]
rf_max_f1 <- rf_long[rf_long$metric == 'Mean F1 Score' & rf_long$value == max(rf$mean_f1), ]

ggplot(rf_long, aes(x=num_features, y=value, color=metric)) +
  geom_line() + 
  scale_color_brewer(palette='Dark2') +
  labs(title='Impact of number of features of random forest performance',
       x='Number of features',
       y='Value',
       color='Metric') +
  geom_point(data = rf_max_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_balanced_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_sensitivity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_specificity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_f1,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3)

# panel generation
ggplot(rf_long, aes(x=num_features, y=value, color=metric)) +
  geom_line() + 
  scale_color_brewer(palette='Dark2') +
  labs(title='B',
       x='Number of features',
       y='Value',
       color='Metric') +
  geom_point(data = rf_max_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_balanced_accuracy,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_sensitivity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_specificity,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) +
  geom_point(data = rf_max_f1,
             mapping = aes(x = num_features, y = value),
             shape = 7,
             size = 3) + 
  theme(legend.position = 'none')

# Performance of different types of models (avgs only) ------

# This, and the section below, are replaced by code incorporating standard deviations below.

diff_models <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_different_models_balanced.csv')
diff_models_slim <- diff_models[c('model_type', 'mean_accuracy', 
                                  'mean_sensitivity', 'mean_specificity',
                                  'mean_balanced_accuracy', 'mean_f1')]
diff_models_slim_long <- pivot_longer(diff_models_slim, cols=!model_type, names_to='metric', values_to='value')
diff_models_slim_long$metric <- str_replace_all(diff_models_slim_long$metric, 
                                  c('mean_accuracy'='Mean Accuracy',
                                    'mean_balanced_accuracy'='Mean Balanced Accuracy',
                                    'mean_sensitivity'='Mean Sensitivity',
                                    'mean_specificity'='Mean Specificity',
                                    'mean_f1'='Mean F1 Score'))

# generate plots
ggplot(diff_models_slim_long, aes(x=model_type, y=value, fill=metric)) +
  geom_bar(stat='identity', position='dodge') +
  labs(title='Performance of different model types',
       x='Model type',
       y='Value',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  ylim(0, 1) + 
  scale_fill_brewer(palette='Dark2')

# generate panels - no legend, title = panel letter. Generate legend by changing
# position to 'bottom'
ggplot(diff_models_slim_long, aes(x=model_type, y=value, fill=metric)) +
  geom_bar(stat='identity', position='dodge') +
  labs(title='B',
       x='Model type',
       y='Value',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  ylim(0, 1) + 
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.position = 'none')

# Effect of balancing data on performance of different models (avgs only) ------
library(tidyverse)
diff_models_balanced <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_different_models_balanced_databymodel.csv')
diff_models_unbalanced <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_different_models.csv')

diff_models_bal_slim <- diff_models_balanced[c('model_type', 'mean_accuracy', 
                                  'mean_sensitivity', 'mean_specificity',
                                  'mean_balanced_accuracy', 'mean_f1')]
diff_models_bal_slim_long <- pivot_longer(diff_models_bal_slim, cols=!model_type, names_to='metric', values_to='value_balanced')
diff_models_bal_slim_long$metric <- str_replace_all(diff_models_bal_slim_long$metric, 
                                                c('mean_accuracy'='Mean Accuracy',
                                                  'mean_balanced_accuracy'='Mean Balanced Accuracy',
                                                  'mean_sensitivity'='Mean Sensitivity',
                                                  'mean_specificity'='Mean Specificity',
                                                  'mean_f1'='Mean F1 Score'))

diff_models_unbal_slim <- diff_models_unbalanced[c('model_type', 'mean_accuracy', 
                                                   'mean_sensitivity', 'mean_specificity',
                                                   'mean_balanced_accuracy', 'mean_f1')]
diff_models_unbal_slim_long <- pivot_longer(diff_models_unbal_slim, cols=!model_type, names_to='metric', values_to='value_unbalanced')
diff_models_unbal_slim_long$metric <- str_replace_all(diff_models_unbal_slim_long$metric, 
                                                    c('mean_accuracy'='Mean Accuracy',
                                                      'mean_balanced_accuracy'='Mean Balanced Accuracy',
                                                      'mean_sensitivity'='Mean Sensitivity',
                                                      'mean_specificity'='Mean Specificity',
                                                      'mean_f1'='Mean F1 Score'))
diff_models_comp <- diff_models_bal_slim_long
diff_models_comp$value_unbalanced <- diff_models_unbal_slim_long$value_unbalanced
diff_models_comp <- diff_models_comp %>%
  mutate(abs_change = value_balanced - value_unbalanced) %>%
  mutate(rel_change =(value_balanced - value_unbalanced)*100/value_unbalanced)

ggplot(diff_models_comp, aes(x=model_type, y=abs_change, fill=metric)) +
  geom_bar(stat='identity', position='dodge') +
  labs(title='Absolute change in model performance \nfollowing dataset balancing',
       x='Model type',
       y='Absolute change',
       fill='Metric') + 
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2')

ggplot(diff_models_comp, aes(x=model_type, y=rel_change, fill=metric)) +
  geom_bar(stat='identity', position='dodge') +
  labs(title='Percentage change in model performance \nfollowing dataset balancing',
       x='Model type',
       y='Percentage change',
       fill='Metric') + 
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2')

# panel generation

ggplot(diff_models_comp, aes(x=model_type, y=abs_change, fill=metric)) +
  geom_bar(stat='identity', position='dodge') +
  labs(title='C',
       x='Model type',
       y='Absolute change',
       fill='Metric') + 
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.position = 'none')

ggplot(diff_models_comp, aes(x=model_type, y=rel_change, fill=metric)) +
  geom_bar(stat='identity', position='dodge') +
  labs(title='D',
       x='Model type',
       y='Percentage change',
       fill='Metric') + 
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.position = 'none')

# Performance of different models + effect of balancing (data from all 100 runs of each model) ------

library(tidyverse)
data_dir <- 'C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/'
model_names = c('Random Forest',
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
               'GPC')

## Unbalanced ------

# how to set up an empty dataframe
unbal_df <- data.frame(matrix(ncol=4, nrow=0))  
colnames(unbal_df) <- c('model_type', 'unbal_mean', 'unbal_sd', 'metric')

# iterates through the data for each model type, gets means and standard
# deviations, puts the data into a plottable form and appends it to the 
# dataframe collating the data from all the models
for (model in model_names) {
  data <- read.csv(paste(data_dir, 'hppy_', model, '_unbalanced.csv', sep=''))
  data_slim <- data[c('accuracy', 
                      'sensitivity', 'specificity',
                      'balanced_accuracy', 'f1')]
  means <- colMeans(data_slim)
  sds <- sapply(data_slim, sd)
  bymodel_df <- data.frame(model_type=rep(model,5), unbal_mean=means, unbal_sd = sds)
  bymodel_df$metric <- row.names(bymodel_df)
  row.names(bymodel_df) <- NULL
  unbal_df <- rbind(unbal_df, bymodel_df)
}

unbal_df$metric <- str_replace_all(unbal_df$metric,
                                   c('accuracy'='Accuracy',
                                     'balanced_Accuracy'='Balanced Accuracy',
                                     'sensitivity'='Sensitivity',
                                     'specificity'='Specificity',
                                     'f1'='F1 Score'))

ggp_unbalanced <- ggplot(unbal_df, aes(x=model_type, y=unbal_mean, fill=metric)) +
  geom_bar(stat='identity', position='dodge', colour='black') +
  # how to do error bars
  geom_errorbar(aes(ymin=unbal_mean-unbal_sd, ymax=unbal_mean+unbal_sd), width=.4, position = position_dodge(.9))+
  labs(title='A',
       x='Model type',
       y='Value',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  ylim(0, 1) +  # y-axis range set so that it matches with the balanced plot next door
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.key.size = unit(5, 'mm')) +  # recommended (by me) practice
  theme(legend.position = 'none')

## Balanced ------
bal_df <- data.frame(matrix(ncol=4, nrow=0))
colnames(bal_df) <- c('model_type', 'bal_mean', 'bal_sd', 'metric')

for (model in model_names) {
  data <- read.csv(paste(data_dir, 'hppy_', model, '_balanced.csv', sep=''))
  data_slim <- data[c('accuracy', 
                      'sensitivity', 'specificity',
                      'balanced_accuracy', 'f1')]
  means <- colMeans(data_slim)
  sds <- sapply(data_slim, sd)
  bymodel_df <- data.frame(model_type=rep(model,5), bal_mean=means, bal_sd = sds)
  bymodel_df$metric <- row.names(bymodel_df)
  row.names(bymodel_df) <- NULL
  bal_df <- rbind(bal_df, bymodel_df)
}

bal_df$metric <- str_replace_all(bal_df$metric,
                                 c('accuracy'='Accuracy',
                                   'balanced_Accuracy'='Balanced Accuracy',
                                   'sensitivity'='Sensitivity',
                                   'specificity'='Specificity',
                                   'f1'='F1 Score'))

ggp_balanced <- ggplot(bal_df, aes(x=model_type, y=bal_mean, fill=metric)) +
  geom_bar(stat='identity', position='dodge', colour='black') +
  geom_errorbar(aes(ymin=bal_mean-bal_sd, ymax=bal_mean+bal_sd), width=.4, position = position_dodge(.9))+
  labs(title='B',
       x='Model type',
       y='Value',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  ylim(0, 1) + 
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.key.size = unit(5, 'mm')) +
  theme(legend.position = 'none')

## Absolute and percentage changes ------

comp_abs_df <- data.frame(matrix(ncol=4, nrow=0))
colnames(comp_abs_df) <- c('model_type', 'abs_mean', 'abs_sd', 'metric')

comp_perc_df <- data.frame(matrix(ncol=4, nrow=0))
colnames(comp_perc_df) <- c('model_type', 'perc_mean', 'perc_sd', 'metric')

for (model in model_names){
  unbal_data <- read.csv(paste(data_dir, 'hppy_', model, '_unbalanced.csv', sep=''))
  bal_data <- read.csv(paste(data_dir, 'hppy_', model, '_balanced.csv', sep=''))
  
  # generate means and standard deviations for absolute changes
  bymodel_abs <- data.frame(accuracy = bal_data$accuracy - unbal_data$accuracy,
                            sensitivity = bal_data$sensitivity - unbal_data$sensitivity,
                            specificity = bal_data$specificity - unbal_data$specificity,
                            balanced_accuracy = bal_data$balanced_accuracy - unbal_data$balanced_accuracy,
                            f1 = bal_data$f1 - unbal_data$f1)
  abs_means <- colMeans(bymodel_abs)
  abs_sds <- sapply(bymodel_abs, sd)
  bymodel_abs_df <- data.frame(model_type=rep(model,5), abs_mean=abs_means, abs_sd = abs_sds)
  bymodel_abs_df$metric <- row.names(bymodel_abs_df)
  row.names(bymodel_abs_df) <- NULL
  comp_abs_df <- rbind(comp_abs_df, bymodel_abs_df)
  
  # generate means and standard deviations for percentage changes
  bymodel_perc <- data.frame(accuracy = (bal_data$accuracy - unbal_data$accuracy)*100/unbal_data$accuracy,
                             sensitivity = (bal_data$sensitivity - unbal_data$sensitivity)*100/unbal_data$sensitivity,
                             specificity = (bal_data$specificity - unbal_data$specificity)*100/unbal_data$specificity,
                             balanced_accuracy = (bal_data$balanced_accuracy - unbal_data$balanced_accuracy)*100/unbal_data$balanced_accuracy,
                             f1 = (bal_data$f1 - unbal_data$f1)*100/unbal_data$f1)
  perc_means <- colMeans(bymodel_perc)
  perc_sds <- sapply(bymodel_perc, sd)
  bymodel_perc_df <- data.frame(model_type=rep(model,5), perc_mean=perc_means, perc_sd = perc_sds)
  bymodel_perc_df$metric <- row.names(bymodel_perc_df)
  row.names(bymodel_perc_df) <- NULL
  comp_perc_df <- rbind(comp_perc_df, bymodel_perc_df)
}

comp_abs_df$metric <- str_replace_all(comp_abs_df$metric,
                                     c('accuracy'='Accuracy',
                                       'balanced_Accuracy'='Balanced Accuracy',
                                       'sensitivity'='Sensitivity',
                                       'specificity'='Specificity',
                                       'f1'='F1 Score'))

comp_perc_df$metric <- str_replace_all(comp_perc_df$metric,
                                      c('accuracy'='Accuracy',
                                        'balanced_Accuracy'='Balanced Accuracy',
                                        'sensitivity'='Sensitivity',
                                        'specificity'='Specificity',
                                        'f1'='F1 Score'))

ggp_abs <- ggplot(comp_abs_df, aes(x=model_type, y=abs_mean, fill=metric)) +
  geom_bar(stat='identity', position='dodge', colour='black') +
  geom_errorbar(aes(ymin=abs_mean-abs_sd, ymax=abs_mean+abs_sd), width=.4, position = position_dodge(.9))+
  labs(title='C',
       x='Model type',
       y='Absolute change',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.key.size = unit(5, 'mm')) +
  theme(legend.position = 'none')

ggp_perc <- ggplot(comp_perc_df, aes(x=model_type, y=perc_mean, fill=metric)) +
  geom_bar(stat='identity', position='dodge', colour='black') +
  geom_errorbar(aes(ymin=perc_mean-perc_sd, ymax=perc_mean+perc_sd), width=.4, position = position_dodge(.9))+
  labs(title='D',
       x='Model type',
       y='Percentage change',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.key.size = unit(5, 'mm')) +
  theme(legend.position = 'none')

## Assemble into four panels with a shared legend ------

library(ggpubr)

ggarrange(
  ggp_unbalanced, ggp_balanced, ggp_abs, ggp_perc, ncol=2,
  nrow=2,
  common.legend = TRUE,
  legend='bottom'
)

# assembling on inkscape looks better imo

## Statistical significance ------

balancing_pval_df <- data.frame(matrix(ncol=6, nrow=0))
colnames(balancing_pval_df) <- c('model_type', 
                       'accuracy_pval',
                       'sensitivty_pval',
                       'specificity_pval',
                       'balanced_accuracy_pval',
                       'f1_pval')

# wilcoxon test used as I neither assume nor think that the values follow a normal distribution
for (model in model_names) {
  bal_data <- read.csv(paste(data_dir, 'hppy_', model, '_balanced.csv', sep=''))
  unbal_data <- read.csv(paste(data_dir, 'hppy_', model, '_unbalanced.csv', sep=''))
  accuracy_pval <- wilcox.test(unbal_data$accuracy, bal_data$accuracy)$p.value
  sensitivity_pval <- wilcox.test(unbal_data$sensitivity, bal_data$sensitivity)$p.value
  specificity_pval <- wilcox.test(unbal_data$sensitivity, bal_data$sensitivity)$p.value
  bal_accuracy_pval <- wilcox.test(unbal_data$balanced_accuracy, bal_data$balanced_accuracy)$p.value
  f1_pval <- wilcox.test(unbal_data$f1, bal_data$f1)$p.value
  bymodel_pvals <- data.frame(model_type=model, 
                              accuracy_pval=accuracy_pval,
                              sensitivity_pval=sensitivity_pval,
                              specificity_pval=specificity_pval,
                              balanced_accuracy_pval=bal_accuracy_pval,
                              f1_pval=f1_pval)
  
  balancing_pval_df <- rbind(balancing_pval_df, bymodel_pvals)
}



# Performance of RFECV models and 90% models vs baselines ------

library(tidyverse)

## LR ------

lr_baseline <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_balanced_baseline_databymodel.csv')

lr_90 <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_balanced_training90_databymodel.csv')

lr_rfe <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_balanced_best_databymodel.csv')

lr_90vsbaseline_abs <- data.frame(accuracy = lr_90$accuracy - lr_baseline$accuracy,
                                  sensitivity = lr_90$sensitivity - lr_baseline$sensitivity,
                                  specificity = lr_90$specificity - lr_baseline$specificity,
                                  balanced_accuracy = lr_90$balanced_accuracy - lr_baseline$balanced_accuracy,
                                  f1 = lr_90$f1 - lr_baseline$f1)
lr_90vsbaseline_abs_means <- colMeans(lr_90vsbaseline_abs)
lr_90vsbaseline_abs_sds <- sapply(lr_90vsbaseline_abs, sd)
lr_90vsbaseline_abs_df <- data.frame(model = rep('lr_90',5), 
                                     abs_mean = lr_90vsbaseline_abs_means,
                                     abs_sd = lr_90vsbaseline_abs_sds)
lr_90vsbaseline_abs_df$metric <- row.names(lr_90vsbaseline_abs_df)
row.names(lr_90vsbaseline_abs_df) <- NULL

lr_90vsbaseline_perc <- data.frame(accuracy = (lr_90$accuracy - lr_baseline$accuracy)*100/lr_baseline$accuracy,
                                   sensitivity = (lr_90$sensitivity - lr_baseline$sensitivity)*100/lr_baseline$sensitivity,
                                   specificity = (lr_90$specificity - lr_baseline$specificity)*100/lr_baseline$specificity,
                                   balanced_accuracy = (lr_90$balanced_accuracy - lr_baseline$balanced_accuracy)*100/lr_baseline$balanced_accuracy,
                                   f1 = (lr_90$f1 - lr_baseline$f1)*100/lr_baseline$f1)
lr_90vsbaseline_perc_means <- colMeans(lr_90vsbaseline_perc)
lr_90vsbaseline_perc_sds <- sapply(lr_90vsbaseline_perc, sd)
lr_90vsbaseline_perc_df <- data.frame(model = rep('lr_90',5), 
                                      perc_mean = lr_90vsbaseline_perc_means,
                                      perc_sd = lr_90vsbaseline_perc_sds)
lr_90vsbaseline_perc_df$metric <- row.names(lr_90vsbaseline_perc_df)
row.names(lr_90vsbaseline_perc_df) <- NULL

lr_rfevsbaseline_abs <- data.frame(accuracy = lr_rfe$accuracy - lr_baseline$accuracy,
                                   sensitivity = lr_rfe$sensitivity - lr_baseline$sensitivity,
                                   specificity = lr_rfe$specificity - lr_baseline$specificity,
                                   balanced_accuracy = lr_rfe$balanced_accuracy - lr_baseline$balanced_accuracy,
                                   f1 = lr_rfe$f1 - lr_baseline$f1)
lr_rfevsbaseline_abs_means <- colMeans(lr_rfevsbaseline_abs)
lr_rfevsbaseline_abs_sds <- sapply(lr_rfevsbaseline_abs, sd)
lr_rfevsbaseline_abs_df <- data.frame(model = rep('lr_rfe',5), 
                                      abs_mean = lr_rfevsbaseline_abs_means,
                                      abs_sd = lr_rfevsbaseline_abs_sds)
lr_rfevsbaseline_abs_df$metric <- row.names(lr_rfevsbaseline_abs_df)
row.names(lr_rfevsbaseline_abs_df) <- NULL

lr_rfevsbaseline_perc <- data.frame(accuracy = (lr_rfe$accuracy - lr_baseline$accuracy)*100/lr_baseline$accuracy,
                                    sensitivity = (lr_rfe$sensitivity - lr_baseline$sensitivity)*100/lr_baseline$sensitivity,
                                    specificity = (lr_rfe$specificity - lr_baseline$specificity)*100/lr_baseline$specificity,
                                    balanced_accuracy = (lr_rfe$balanced_accuracy - lr_baseline$balanced_accuracy)*100/lr_baseline$balanced_accuracy,
                                    f1 = (lr_rfe$f1 - lr_baseline$f1)*100/lr_baseline$f1)
lr_rfevsbaseline_perc_means <- colMeans(lr_rfevsbaseline_perc)
lr_rfevsbaseline_perc_sds <- sapply(lr_rfevsbaseline_perc, sd)
lr_rfevsbaseline_perc_df <- data.frame(model = rep('lr_rfe',5), 
                                       perc_mean = lr_rfevsbaseline_perc_means,
                                       perc_sd = lr_rfevsbaseline_perc_sds)
lr_rfevsbaseline_perc_df$metric <- row.names(lr_rfevsbaseline_perc_df)
row.names(lr_rfevsbaseline_perc_df) <- NULL

## RF ------

rf_baseline <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_rf_balanced_baseline_databymodel.csv')

rf_90 <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_rf_balanced_training90_databymodel.csv')

rf_rfe <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_rf_balanced_best_databymodel.csv')

rf_90vsbaseline_abs <- data.frame(accuracy = rf_90$accuracy - rf_baseline$accuracy,
                                  sensitivity = rf_90$sensitivity - rf_baseline$sensitivity,
                                  specificity = rf_90$specificity - rf_baseline$specificity,
                                  balanced_accuracy = rf_90$balanced_accuracy - rf_baseline$balanced_accuracy,
                                  f1 = rf_90$f1 - rf_baseline$f1)
rf_90vsbaseline_abs_means <- colMeans(rf_90vsbaseline_abs)
rf_90vsbaseline_abs_sds <- sapply(rf_90vsbaseline_abs, sd)
rf_90vsbaseline_abs_df <- data.frame(model = rep('rf_90',5), 
                                    abs_mean = rf_90vsbaseline_abs_means,
                                    abs_sd = rf_90vsbaseline_abs_sds)
rf_90vsbaseline_abs_df$metric <- row.names(rf_90vsbaseline_abs_df)
row.names(rf_90vsbaseline_abs_df) <- NULL

rf_90vsbaseline_perc <- data.frame(accuracy = (rf_90$accuracy - rf_baseline$accuracy)*100/rf_baseline$accuracy,
                                   sensitivity = (rf_90$sensitivity - rf_baseline$sensitivity)*100/rf_baseline$sensitivity,
                                   specificity = (rf_90$specificity - rf_baseline$specificity)*100/rf_baseline$specificity,
                                   balanced_accuracy = (rf_90$balanced_accuracy - rf_baseline$balanced_accuracy)*100/rf_baseline$balanced_accuracy,
                                   f1 = (rf_90$f1 - rf_baseline$f1)*100/rf_baseline$f1)
rf_90vsbaseline_perc_means <- colMeans(rf_90vsbaseline_perc)
rf_90vsbaseline_perc_sds <- sapply(rf_90vsbaseline_perc, sd)
rf_90vsbaseline_perc_df <- data.frame(model = rep('rf_90',5), 
                                      perc_mean = rf_90vsbaseline_perc_means,
                                      perc_sd = rf_90vsbaseline_perc_sds)
rf_90vsbaseline_perc_df$metric <- row.names(rf_90vsbaseline_perc_df)
row.names(rf_90vsbaseline_perc_df) <- NULL

rf_rfevsbaseline_abs <- data.frame(accuracy = rf_rfe$accuracy - rf_baseline$accuracy,
                                  sensitivity = rf_rfe$sensitivity - rf_baseline$sensitivity,
                                  specificity = rf_rfe$specificity - rf_baseline$specificity,
                                  balanced_accuracy = rf_rfe$balanced_accuracy - rf_baseline$balanced_accuracy,
                                  f1 = rf_rfe$f1 - rf_baseline$f1)
rf_rfevsbaseline_abs_means <- colMeans(rf_rfevsbaseline_abs)
rf_rfevsbaseline_abs_sds <- sapply(rf_rfevsbaseline_abs, sd)
rf_rfevsbaseline_abs_df <- data.frame(model = rep('rf_rfe',5), 
                                     abs_mean = rf_rfevsbaseline_abs_means,
                                     abs_sd = rf_rfevsbaseline_abs_sds)
rf_rfevsbaseline_abs_df$metric <- row.names(rf_rfevsbaseline_abs_df)
row.names(rf_rfevsbaseline_abs_df) <- NULL

rf_rfevsbaseline_perc <- data.frame(accuracy = (rf_rfe$accuracy - rf_baseline$accuracy)*100/rf_baseline$accuracy,
                                   sensitivity = (rf_rfe$sensitivity - rf_baseline$sensitivity)*100/rf_baseline$sensitivity,
                                   specificity = (rf_rfe$specificity - rf_baseline$specificity)*100/rf_baseline$specificity,
                                   balanced_accuracy = (rf_rfe$balanced_accuracy - rf_baseline$balanced_accuracy)*100/rf_baseline$balanced_accuracy,
                                   f1 = (rf_rfe$f1 - rf_baseline$f1)*100/rf_baseline$f1)
rf_rfevsbaseline_perc_means <- colMeans(rf_rfevsbaseline_perc)
rf_rfevsbaseline_perc_sds <- sapply(rf_rfevsbaseline_perc, sd)
rf_rfevsbaseline_perc_df <- data.frame(model = rep('rf_rfe',5), 
                                      perc_mean = rf_rfevsbaseline_perc_means,
                                      perc_sd = rf_rfevsbaseline_perc_sds)
rf_rfevsbaseline_perc_df$metric <- row.names(rf_rfevsbaseline_perc_df)
row.names(rf_rfevsbaseline_perc_df) <- NULL

## Statistical significance ------

lr_90vsbaseline_accuracy_pval <- wilcox.test(lr_baseline$accuracy, lr_90$accuracy)$p.val
lr_90vsbaseline_sensitivity_pval <- wilcox.test(lr_baseline$sensitivity, lr_90$sensitivity)$p.val
lr_90vsbaseline_specificity_pval <- wilcox.test(lr_baseline$specificity, lr_90$specificity)$p.val
lr_90vsbaseline_balanced_accuracy_pval <- wilcox.test(lr_baseline$balanced_accuracy, lr_90$balanced_accuracy)$p.val
lr_90vsbaseline_f1_pval <- wilcox.test(lr_baseline$f1, lr_90$f1)$p.val

lr_rfevsbaseline_accuracy_pval <- wilcox.test(lr_baseline$accuracy, lr_rfe$accuracy)$p.val
lr_rfevsbaseline_sensitivity_pval <- wilcox.test(lr_baseline$sensitivity, lr_rfe$sensitivity)$p.val
lr_rfevsbaseline_specificity_pval <- wilcox.test(lr_baseline$specificity, lr_rfe$specificity)$p.val
lr_rfevsbaseline_balanced_accuracy_pval <- wilcox.test(lr_baseline$balanced_accuracy, lr_rfe$balanced_accuracy)$p.val
lr_rfevsbaseline_f1_pval <- wilcox.test(lr_baseline$f1, lr_rfe$f1)$p.val

rf_90vsbaseline_accuracy_pval <- wilcox.test(rf_baseline$accuracy, rf_90$accuracy)$p.val
rf_90vsbaseline_sensitivity_pval <- wilcox.test(rf_baseline$sensitivity, rf_90$sensitivity)$p.val
rf_90vsbaseline_specificity_pval <- wilcox.test(rf_baseline$specificity, rf_90$specificity)$p.val
rf_90vsbaseline_balanced_accuracy_pval <- wilcox.test(rf_baseline$balanced_accuracy, rf_90$balanced_accuracy)$p.val
rf_90vsbaseline_f1_pval <- wilcox.test(rf_baseline$f1, rf_90$f1)$p.val

rf_rfevsbaseline_accuracy_pval <- wilcox.test(rf_baseline$accuracy, rf_rfe$accuracy)$p.val
rf_rfevsbaseline_sensitivity_pval <- wilcox.test(rf_baseline$sensitivity, rf_rfe$sensitivity)$p.val
rf_rfevsbaseline_specificity_pval <- wilcox.test(rf_baseline$specificity, rf_rfe$specificity)$p.val
rf_rfevsbaseline_balanced_accuracy_pval <- wilcox.test(rf_baseline$balanced_accuracy, rf_rfe$balanced_accuracy)$p.val
rf_rfevsbaseline_f1_pval <- wilcox.test(rf_baseline$f1, rf_rfe$f1)$p.val

## Plotting ------

abs_data <- rbind(rf_90vsbaseline_abs_df, rf_rfevsbaseline_abs_df,
                  lr_90vsbaseline_abs_df, lr_rfevsbaseline_abs_df)

abs_data$metric <- str_replace_all(abs_data$metric,
                                   c('accuracy'='Accuracy',
                                   'balanced_Accuracy'='Balanced Accuracy',
                                   'sensitivity'='Sensitivity',
                                   'specificity'='Specificity',
                                   'f1'='F1 Score'))

ggplot(abs_data, aes(x=model, y=abs_mean, fill=metric)) + 
  geom_bar(stat='identity', position='dodge', colour='black') +
  geom_errorbar(aes(ymin=abs_mean-abs_sd, ymax=abs_mean+abs_sd), width=.4, position = position_dodge(.9)) +
  labs(title='C',
       x='Model',
       y='Absolute change in performance of \n specified model compared to baseline',
       fill='Metric') +
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.key.size = unit(5, 'mm')) +
  theme(legend.position = 'none')

perc_data <- rbind(rf_90vsbaseline_perc_df, rf_rfevsbaseline_perc_df,
                   lr_90vsbaseline_perc_df, lr_rfevsbaseline_perc_df)

perc_data$metric <- str_replace_all(perc_data$metric,
                                    c('accuracy'='Accuracy',
                                      'balanced_Accuracy'='Balanced Accuracy',
                                      'sensitivity'='Sensitivity',
                                      'specificity'='Specificity',
                                      'f1'='F1 Score'))

ggplot(perc_data, aes(x=model, y=perc_mean, fill=metric)) + 
  geom_bar(stat='identity', position='dodge', colour='black') +
  geom_errorbar(aes(ymin=perc_mean-perc_sd, ymax=perc_mean+perc_sd), width=.4, position = position_dodge(.9)) +
  labs(title='D',
       x='Model',
       y='Percentage change in performance of \n specified model compared to baseline',
       fill='Metric') +
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.position = 'none')

# Feature importances ------

library(tidyverse)
fingerprints <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/pubchem_fingerprints.csv')

## LR overall -----

lr <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_balanced_best_databymodel.csv')
lr <- lr[,8:105]
lr_means <- colMeans(lr)
lr_sds <- sapply(lr, sd)

# make dataframe from mean and s.d. data, with empty columns to be filled by confidence interval boundaries
lr_df <- data.frame(means=lr_means, sds=lr_sds)
lr_df$ci_low <- 0
lr_df$ci_high <- 0

# use wilcoxon single sign test to get the 95% confidence intervals 
for (i in 1:length(colnames(lr))) {
  wilcox <- wilcox.test(lr[,i], conf.int=TRUE, conf.level=0.95)
  cis <- wilcox$conf.int
  ci_low <- cis[1]
  ci_high <- cis[2]
  lr_df[i, 3] <- ci_low
  lr_df[i, 4] <- ci_high
}

# wizardry to get feature names (PubchemFPx) and their corresponding fingerprint 
lr_df$feature <- row.names(lr_df)
lr_df$feature <- str_remove_all(lr_df$feature, '_gini.coefs')
lr_df$bit <- str_remove_all(lr_df$feature, 'PubchemFP')
lr_df <- merge(lr_df, fingerprints, by='bit')

# preprae for ranking of features and colouring of features by their sign
lr_df <- lr_df %>%
  mutate(abs_means = abs(means)) %>%
  mutate(value_signs = case_when(means < 0 ~ 'Negative',
                                 means > 0 ~ 'Positive',
                                 means == 0 ~ 'Zero'))

# get the top 10 features 
lr_ordered <- lr_df[order(-lr_df$abs_means), ]
lr_top10 <- lr_ordered[1:10,]
lr_top10 <- lr_top10[order(lr_top10$abs_means), ]
lr_top10$bit <- factor(lr_top10$bit, levels = lr_top10$bit)
lr_top10$value_signs <- factor(lr_top10$value_signs, levels = c('Positive', 'Negative'))

# plot the top 10 features and their log-odds ratio on a rotated axis (bars go horizontally)
ggplot(lr_top10, aes(x=bit, y=means, fill=value_signs)) + 
  geom_bar(stat='identity', colour='black') + 
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high, width=.5)) + 
  labs(title = 'E',
       x = 'Fingerprint',
       y = 'log(odds ratio)',
       fill = 'Predicted impact on inhibitory activity') + 
  scale_fill_brewer(palette='Dark2') + 
  theme(legend.key.size = unit(5, 'mm')) +
  coord_flip() + 
  theme(legend.position = 'none')

## LR individual ------

# no C.I.s needed here
lr <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_lr_balanced_best_databymodel.csv')
lr_indiv <- lr[lr$random_state == 380, ]
lr_indiv <- lr_indiv[, 8:105]
lr_indiv_long <- pivot_longer(lr_indiv, cols = colnames(lr_indiv), names_to='feature', values_to='odds')
lr_indiv_long$feature <- str_remove_all(lr_indiv_long$feature, '_gini.coefs')
lr_indiv_long$bit <- str_remove_all(lr_indiv_long$feature, 'PubchemFP')
lr_indiv_df <- merge(lr_indiv_long, fingerprints, by='bit')
lr_indiv_df <- lr_indiv_df %>%
  mutate(abs_odds = abs(odds)) %>%
  mutate(value_signs = case_when(odds < 0 ~ 'Negative',
                                 odds > 0 ~ 'Positive',
                                 odds == 0 ~ 'Zero'))

lr_indiv_ordered <- lr_indiv_df[order(-lr_indiv_df$abs_odds), ]
lr_indiv_top10 <- lr_indiv_ordered[1:10, ]
lr_indiv_top10 <- lr_indiv_top10[order(lr_indiv_top10$abs_odds), ]
lr_indiv_top10$bit <- factor(lr_indiv_top10$bit, levels = lr_indiv_top10$bit)
lr_indiv_top10$value_signs <- factor(lr_indiv_top10$value_signs, levels = c('Positive', 'Negative'))

ggplot(lr_indiv_top10, aes(x=bit, y=odds, fill=value_signs)) + 
  geom_bar(stat='identity', colour='black') +
  labs(title = 'E',
       x = 'Fingerprint',
       y = 'log(odds ratio)',
       fill = 'Predicted impact on inhibitory activity') + 
  scale_fill_brewer(palette='Dark2') + 
  coord_flip() + 
  theme(legend.position = 'none')

# Performance of different DNN architectures and train-val-test splits ------

library(tidyverse)

model_ids <- c('60-20-20_v1 \n(DNN1)',
               '60-20-20_v2',
               '60-20-20_v3',
               '70-15-15_v1 \n(DNN2)',
               '70-15-15_v2',
               '80-10-10_v1',
               '80-10-10_v2 \n(DNN3)')

tuneds <- data.frame(matrix(ncol=4, nrow=0))
colnames(tuneds) <- c('means', 'sds', 'metric', 'id')

for (i in 1:7){
  tuned_model <- data.frame(matrix(ncol=5, nrow=0))
  colnames(tuned_model) <- c('accuracy', 
                             'sensitivity', 'specificity',
                             'balanced_accuracy', 'f1')
  model <- paste('tuned', as.character(i), sep='')
  for (j in 1:3) {
    run_num <- paste('run', as.character(j), sep='')
    filename <- paste(model, run_num, sep='_')
    tuned_run <- read.csv(paste('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/dnn_colab_data/dnn_', 
                                model, '/', run_num, '/dnn_', filename, '_test_stats.csv', sep=''))
    tuned_model <- rbind(tuned_model, tuned_run)
  }
  model_means <- data.frame(means=colMeans(tuned_model), sds=sapply(tuned_model, sd))
  model_means$metric <- c('accuracy', 
                          'sensitivity', 'specificity',
                          'balanced_accuracy', 'f1')
  model_means$id <- model_ids[i]
  tuneds <- rbind(tuneds, model_means)
}

row.names(tuneds) <- NULL 

tuneds$metric <- str_replace_all(tuneds$metric,
                                 c('accuracy'='Accuracy',
                                   'balanced_Accuracy'='Balanced Accuracy',
                                   'sensitivity'='Sensitivity',
                                   'specificity'='Specificity',
                                   'f1'='F1 Score'))

ggplot(tuneds, aes(x=id, y=means, fill=metric)) + 
  geom_bar(stat='identity', position='dodge', colour='black') + 
  geom_errorbar(aes(ymin=means-sds, ymax=means+sds, width=.4), position=position_dodge(.9)) + 
  labs(title='B',
       x='Model',
       y='Value',
       fill='Metric') +
  theme(axis.text.x = element_text(angle=45, hjust=0.95, vjust=1)) +
  scale_fill_brewer(palette='Dark2') +
  theme(legend.key.size = unit(5, 'mm')) +
  theme(legend.position = 'bottom')

tuneds_wide <- pivot_wider(tuneds, values_from=c('means', 'sds'), names_from=metric)

# Train-val losses over epoch time -------

library(tidyverse)

counter <- 0

# generates nine ggplot variables, each containing the traces for the 3 runs of each of the 3 models
for (i in c(1, 4, 7)) {
  model <- paste('tuned', as.character(i), sep='')
  for (j in 1:3) {
    counter <- counter + 1
    run_num <- paste('run', as.character(j), sep='')
    filename <- paste(model, run_num, sep='_')
    
    historiae <- data.frame(matrix(ncol=4, nrow=0))
    colnames(historiae) <- c('epoch', 'run', 'which_loss', 'value')
    
    # iterate through the training-validation histories to prepare them for plotting 
    # and append to the collating dataframe
    for (k in 1:100){
      historia <- read.csv(paste('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/dnn_colab_data/dnn_', 
                                 model, '/', run_num, '/dnn_', filename, 
                                 '_training_val_history_', as.character(k-1), 
                                 '.csv', sep=''))
      historia$epoch <- as.numeric(row.names(historia))
      historia$run <- k
      historia <- historia[c('loss', 'val_loss', 'epoch', 'run')]
      historia_long <- pivot_longer(historia, cols=c('loss', 'val_loss'), names_to='which_loss', values_to='value')
      historiae <- rbind(historiae, historia_long)
    }
    
    # separate the loss types - necessary for plotting 
    historiae_loss <- historiae[historiae$which_loss == 'loss', ]
    historiae_val_loss <- historiae[historiae$which_loss == 'val_loss', ]
    
    ggp <- ggplot() +
      geom_line(data=historiae_loss, mapping=aes(x=epoch, y=value, group=factor(run), colour='blue')) +
      geom_line(data=historiae_val_loss, mapping=aes(x=epoch, y=value, group=factor(run), colour='red')) +
      labs(x='Epoch',
           y='Value',
           colour='Loss') +
      scale_color_discrete(labels=c('Training \nLoss', 'Validation \nLoss'))  # changes the label for the losses from the specified colours to those loss names
    
    if (counter == 1) {
      ggp1 <- ggp
    }
    else if (counter == 2) {
      ggp2 <- ggp
    }
    else if (counter == 3) {
      ggp3 <- ggp
    }
    else if (counter == 4) {
      ggp4 <- ggp
    }
    else if (counter == 5) {
      ggp5 <- ggp
    }
    else if (counter == 6) {
      ggp6 <- ggp
    }
    else if (counter == 7) {
      ggp7 <- ggp
    }
    else if (counter == 8) {
      ggp8 <- ggp
    }
    else if (counter == 9) {
      ggp9 <- ggp
    }
  }
}

# Collate the 9 trajectory plots into a single 3x3 grid, with a shared legend
ggarrange(ggp1, ggp2, ggp3, ggp4, ggp5, ggp6, ggp7, ggp8, ggp9,
          nrow=3, ncol=3,
          common.legend = TRUE,
          legend = 'bottom')

# Table for predictions ------

library(tidyverse)
library(gt)

predictions <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/happyhour_ukb_predicted_inhibitors.csv', check.names=FALSE)
colnames(predictions)[1] <- 'Predictions'

gt(predictions) %>%
  opt_table_font('Helvetica') %>%
  tab_style(  # change cells with Y to green, and hide the text by changing it to the same colour as the cell fill
    style = list(
      cell_fill(color = '#66A61E'),
      cell_text(color = '#66A61E')
    ),
    locations = list(
      cells_body(
        columns = RF_678,
        rows = RF_678 == 'Y'),
      cells_body(
        columns = RF_981,
        rows = RF_981 == 'Y'),
      cells_body(
        columns = LR_380,
        rows = LR_380 == 'Y'),
      cells_body(
        columns = LR_722,
        rows = LR_722 == 'Y'),
      cells_body(
        columns = DNN1,
        rows = DNN1 == 'Y'),
      cells_body(
        columns = DNN2,
        rows = DNN2 == 'Y'),
      cells_body(
        columns = DNN3,
        rows = DNN3 == 'Y')
    )
  ) %>%
  tab_style(  # indicate which predictions are taken forward
    style = cell_fill(color = '#E6AB02'),
    locations = cells_body(
      columns = Predictions,
      rows = Predictions %in% c('Losartan', 'Tegretol', 'Salamol (Salbutamol)')
    )
  ) %>%
  tab_style(  # add grey vertical borders to the main body of the table and to the column headers
    style = cell_borders(
      sides = c('left', 'right'),
      color = '#D3D3D3'
    ),
    locations = list(cells_body(), cells_column_labels())
  ) %>%
  tab_style(  # make the column labels bold
    style = cell_text(
      weight = 'bold'
    ),
    locations = cells_column_labels()
  ) %>%
  tab_style(  # add darker vertical lines between the model types in the main body and column headers
    style = cell_borders(
      sides = 'right',
      color = '#808080'
    ),
    locations = list(cells_body(
      columns = c('Predictions', 'RF_981', 'LR_722', 'DNN3')
    ), cells_column_labels(
      columns = c('Predictions', 'RF_981', 'LR_722', 'DNN3')
    ))
  ) %>%
  tab_style(  # make binding energy values aligned centrally
    style = cell_text(
      align = 'center'
    ),
    locations = cells_body(
      columns = 'Binding energies (kcal/mol)'
    )
  )

# Table for DNNs ------

library(tidyverse)
library(gt)

constructions <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/hppy_dnn_tuned_constructions.csv', check.names=FALSE)

gt(constructions) %>%
  opt_table_font('Helvetica') %>%
  tab_style(  # add grey vertical borders to the main body of the table
    style = list(cell_borders(
      sides = c('left', 'right'),
      color = '#D3D3D3'
    ), cell_text(
      align = 'center'
      )),
    locations = list(cells_body(), cells_column_labels())
  ) %>%
  tab_style(  # make the column labels bold
    style = cell_text(
      weight = 'bold'
    ),
    locations = cells_column_labels()
  )%>%
  tab_style(  # make binding energy values aligned centrally
    style = cell_text(
      align = 'center'
    ),
    locations = list(cells_body(), cells_column_labels())
  )

# Table for classification criteria ------

criteria <- read.csv('C:/Users/krish/PycharmProjects/happyhour_inhibition_ml/data/happyhour_bioactivity_class_criteria.csv', check.names=FALSE)

colnames(criteria) <- c("Data Type", "Conditions for 'inhibitory'", "Conditions for 'non-inhibitory'")
criteria[1,2] <- "Data given as '=x' or '<x'"
criteria[1,3] <- "Data given as '>x'"
criteria[2,2] <- "<= 65"
criteria[3,2] <- ">= 70"


gt(criteria) %>%
  opt_table_font('Helvetica') %>%
  tab_style(  # add grey vertical borders to the main body of the table
    style = list(cell_borders(
      sides = c('left', 'right'),
      color = '#D3D3D3'
    ), cell_text(
      align = 'center'
    )),
    locations = list(cells_body(), cells_column_labels())
  ) %>%
  tab_style(  # make the column labels bold
    style = cell_text(
      weight = 'bold'
    ),
    locations = cells_column_labels()
  )%>%
  tab_style(  # make binding energy values aligned centrally
    style = cell_text(
      align = 'center'
    ),
    locations = list(cells_body(), cells_column_labels())
  )

