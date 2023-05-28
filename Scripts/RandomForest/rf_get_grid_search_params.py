import os, random, math, time, uuid
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

# Declare Config Params
n_jobs = 32 - 1
n_splits = 2
n_repeats = 1
random_state = 42

impute = True
strategy = 'mean'
normalise = True
standardise = True

hyper_params_grid = dict()
hyper_params_grid['max_depth'] = [200]
hyper_params_grid['n_estimators'] = [300]
hyper_params_grid['max_features'] = [0.3]
hyper_params_grid['max_samples'] = [0.9]
hyper_params_grid['min_samples_split'] = [40]
hyper_params_grid['bootstrap'] = [True]

# Declare Grid Search Ranges 
max_depth_step = 1
max_depth_min = 1
max_depth_max = 40 + max_depth_step

# max_depth: the maximum depth of the tree 
hyper_params_grid['max_depth'] = np.arange(
    max_depth_min, max_depth_max, max_depth_step
)

# n_estimators: the number of trees in the forest
num_trees_step = 10
num_trees_min = 10
num_trees_max = 150 + num_trees_step

hyper_params_grid['n_estimators'] = np.arange(
    num_trees_min, num_trees_max, num_trees_step
)

max_features_step = .05
max_features_min = .1
max_features_max = .9 + max_features_step

hyper_params_grid['max_features'] = np.arange(
    max_features_min, max_features_max, max_features_step
)

max_samples_step = .05
max_samples_min = .1
max_samples_max = .9 + max_samples_step

hyper_params_grid['max_samples'] = np.arange(
    max_samples_min, max_samples_max, max_samples_step
)

min_samples_split_step = 1
min_samples_split_min = 5
min_samples_split_max = 40 + min_samples_split_step

hyper_params_grid['min_samples_split'] = np.arange(
    min_samples_split_min, min_samples_split_max, min_samples_split_step
)

hyper_params_grid['bootstrap'] = [True]

source_filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
grid_results_filename = 'Analysis/RandomForest/grid_search_parmas.csv'.format(date_str)

# Begin Core Script...
# Import Data
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos = phenos[phenos_subset[:1:3]]

# Standardise Data
scores = scores['f_kir_score'].values.reshape(-1,1)
if len(phenos.values.shape) == 1:
    phenos = phenos.values.reshape(-1,1)
else:
    phenos = phenos.values[:, 0:]

phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
    X=phenos, Y=scores, impute = impute, standardise = standardise, 
    normalise = normalise, strategy=strategy
)

scores = scores.ravel()

# Instantiate Model & Hyper Params
cv = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=random_state
)

gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=hyper_params_grid,
    cv=cv, 
    scoring='neg_mean_absolute_error', 
    verbose=0, 
    n_jobs=n_jobs # parallelism, -1 means using all processors
)

# Perform Grid Search
grid_search_results = gsc.fit(phenos, scores) # Regress Immunos on iKIR Score
h_params = grid_search_results.best_params_
avg_neg_mae = grid_search_results.best_score_

#Export Grid Search Results 
h_params_sets = grid_search_results.cv_results_['params']
neg_mae_scores = grid_search_results.cv_results_['mean_test_score']
h_param_labels = ['max_depth', 'n_estimators', 'max_features', 'max_samples', 'bootstrap', 'min_samples_split']

grid_results = [[h_params_sets[i][key] for key in h_param_labels] \
                    for i in range(0, len(h_params_sets),1)
]
grid_results = pd.DataFrame(grid_results, columns=h_param_labels)
grid_results.to_csv(grid_results_filename)
print('Complete.')