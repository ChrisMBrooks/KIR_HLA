# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from Controllers.DataScienceManager import DataScienceManager as dsm

def preprocess(phenos:pd.DataFrame, scores:pd.DataFrame):
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
    return phenos, scores

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

#Read in Subset of Immunophenotypes
cut_off = 10
filename = "Data/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(cut_off))
phenos_subset = pd.read_csv(filename, index_col=0)
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])
phenos_subset = ['MFI:469', 'P1:20102', 'P1:4229', 'P1:20709']

scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_t = phenos_t[phenos_subset]

scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos_v = phenos_v[phenos_subset]

# Standardise Data
phenos_t, scores_t = preprocess(phenos_t, scores_t)
phenos_v, scores_v = preprocess(phenos_v, scores_v)

# Fit the Model 
n_splits = 4
n_repeats = 100
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

max_depth = 2
n_estimators = 90

model = RandomForestRegressor(
    max_depth=max_depth, 
    n_estimators=n_estimators,
    bootstrap=True,
    max_samples=0.8,
    random_state=False, 
    verbose=1
)

fitted_model = model.fit(phenos_t, scores_t)

num_repeats = 10
results = permutation_importance(fitted_model, phenos_v, scores_v, n_repeats=num_repeats,
    random_state=0, scoring = 'neg_mean_absolute_error', n_jobs=-1
)

results_stats = [] 

for i in results.importances_mean.argsort()[::-1]:
    results_stats.append([phenos_subset[i], results.importances_mean[i], results.importances_std[i]])

results_stats = pd.DataFrame(results_stats, columns = ['feature', 'importance_mean', 'importance_std'])

results_stats = results_stats.sort_values(by='importance_mean', ascending=False)
filename = 'Analysis/RandomForest/feature_importance_perm_rankings_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())

results_stats.to_csv(filename)

print('Complete.')

"""

The permutation feature importance is the decrease in a model score when a single feature value is randomly shuffled.

Permutation importances can be computed either on the training set or on a held-out testing or validation set. 
Using a held-out set makes it possible to highlight which features contribute the most to 
the generalization power of the inspected model.

"""