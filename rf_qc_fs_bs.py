# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Set Script Config
if len(sys.argv) == 5:
    print('Setting Sys Params:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[3])
    count = int(sys.argv[1])
    forward_selection = bool(sys.argv[2])
    backward_selection = bool(sys.argv[3])
    iteration_id = int(sys.argv[4])
else:
    count = 100
    forward_selection = True
    backward_selection = True
    iteration_id = 1

use_full_dataset=True
use_database = False

#Instantiate Controllers
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

# Declare Config Params
n_jobs = 16 - 1
n_splits = 4
n_repeats = 10
random_state_1 = 42
random_state_2 = 21
tolerance = None

impute = True
standardise = False
normalise = False
strategy='mean'

h_params = dict()

h_params['max_depth'] = 200
h_params['n_estimators'] = 300
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['bootstrap'] = True

#Read in Subset of Immunophenotypes
""" cut_off = 0.0001
filename = "Analysis/RandomForest/feature_importance_impurity_rankings_13042023.csv"
phenos_subset = pd.read_csv(filename, index_col=0)
phenos_subset = phenos_subset[phenos_subset > cut_off].dropna()
phenos_subset = list(phenos_subset.index) """

filename = "Analysis/RandomForest/17042023/feature_importance_impurity_rankings_17042023_{}.csv".format(iteration_id)
phenos_subset = pd.read_csv(filename, index_col=0)
phenos_subset = phenos_subset.sort_values(by='0', axis=0, ascending=False)
phenos_subset = phenos_subset.index[0:count]

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos = phenos[phenos_subset]

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

# Fit the Model 
cv = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=random_state_1
)

# Instantiate Model    
model = RandomForestRegressor(
    max_depth=h_params["max_depth"], 
    n_estimators=h_params["n_estimators"],
    bootstrap=h_params["bootstrap"],
    max_features=h_params["max_features"],
    max_samples=h_params["max_samples"],
    random_state=random_state_2, 
    verbose=1,
    n_jobs=n_jobs
)

if forward_selection:
    print('Starting Forward Selection...')
    sfs_for = SequentialFeatureSelector(
        model, direction='forward', 
        n_features_to_select='auto', scoring='neg_mean_absolute_error', 
        tol=tolerance, cv=cv, n_jobs=n_jobs
    )

    sfs_for.fit(phenos, scores)
    for_selected_features = sfs_for.get_support()
    print('Forward Selection Complete.')

if backward_selection:
    print('Starting Backward Selection...')
    sfs_bac = SequentialFeatureSelector(
        model, direction='backward', 
        n_features_to_select='auto', scoring='neg_mean_absolute_error', 
        tol=tolerance, cv=cv, n_jobs=n_jobs
    )
    sfs_bac.fit(phenos, scores)
    bac_selected_features = sfs_bac.get_support()
    print('Backward Selection Complete.')

print('Exporting Results...')
flag = ''
if forward_selection and not backward_selection:
    flag = 'fs'
    summary = [[phenos_subset[i], for_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
    summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected'])
elif backward_selection and not forward_selection:
    flag = 'bs'
    summary = [[phenos_subset[i], bac_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
    summary_df = pd.DataFrame(summary, columns=['label', 'backward_selected'])
else:
    flag='fs_bs'
    summary = [[phenos_subset[i], for_selected_features[i], bac_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
    summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected', 'backward_selected'])

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/RandomForest/r_forest_{}_candidate_features_{}_{}_{}.csv".format(flag, count, date_str, iteration_id)
summary_df.to_csv(filename)
print('Complete.')

