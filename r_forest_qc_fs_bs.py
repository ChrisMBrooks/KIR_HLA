# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

#Read in Subset of Immunophenotypes
cut_off = 0
filename = "Analysis/RandomForest/feature_importance_rankings_26032023.csv"
phenos_subset = pd.read_csv(filename, index_col=0)
phenos_subset = phenos_subset[phenos_subset > cut_off].copy()
phenos_subset = list(phenos_subset.index)

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
    X=phenos, Y=scores, impute = True, standardise = False, 
    normalise = True
)

scores = scores.ravel()

# Fit the Model 
n_splits = 2
n_repeats = 2
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

max_depth = 15
n_estimators = 460

model = RandomForestRegressor(
    max_depth=max_depth, 
    n_estimators=n_estimators,
    bootstrap=True,
    max_samples=0.8,
    random_state=False, 
    verbose=1
)

sfs_for = SequentialFeatureSelector(model, direction='forward', scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
sfs_for.fit(phenos, scores)
print('Forward Selection Complete.')

sfs_bac = SequentialFeatureSelector(model, direction='backward', scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
sfs_bac.fit(phenos, scores)
print('Backward Selection Complete.')

for_selected_features = sfs_for.get_support()
bac_selected_features = sfs_bac.get_support()

summary = [[phenos_subset[i], for_selected_features[i], bac_selected_features[i]] for i in range(0, phenos_subset.shape[0], 1)]

summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected', 'backward_selected'])

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/RandomForest/r_forest_fs_bs_candidate_features_{}.csv".format(date_str)
summary_df.to_csv(filename)
print('Complete.')
