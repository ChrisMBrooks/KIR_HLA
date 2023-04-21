# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, random, math
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from Controllers.DataScienceManager import DataScienceManager as dsm

def partition_dataframes(phenos:pd.DataFrame, scores:pd.DataFrame, ratio:float):
    indeces = list(range(0, phenos.shape[0]))
    random.shuffle(indeces)
    cutoff = math.floor(ratio*phenos.shape[0])
    subset_indeces = indeces[:cutoff]
    reaminder_indeces = indeces[cutoff:]

    phenos_df_tt = phenos.iloc[subset_indeces, :]
    scores_df_tt = scores.iloc[subset_indeces, :]

    phenos_df_tv = phenos.iloc[reaminder_indeces, :]
    scores_df_tv = scores.iloc[reaminder_indeces, :]
    return phenos_df_tt, scores_df_tt, phenos_df_tv, scores_df_tv

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
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

segregate_dataset = True
num_repeats = 10
n_jobs = 16 - 1
random_state = 42

#Read in Subset of Immunophenotypes

filename = 'Analysis/RandomForest/14042023_100/r_forest_fs_bs_candidate_features_100_14042023.csv'
phenos_subset = pd.read_csv(filename, index_col=0)
phenos_subset['summation'] = phenos_subset['forward_selected'] + phenos_subset['backward_selected']
phenos_subset = phenos_subset[phenos_subset['summation'] > 0]
phenos_subset = list(phenos_subset['label'].values)

scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_t = phenos_t[phenos_subset]

scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_v = phenos_v[phenos_subset]

if segregate_dataset:
    phenos_t, scores_t, phenos_v, scores_v = partition_dataframes(phenos_t, scores_t, 0.8)

# Standardise Data
phenos_t, scores_t = preprocess(phenos_t, scores_t)
phenos_v, scores_v = preprocess(phenos_v, scores_v)

h_params = dict()
h_params['max_depth'] = 6
h_params['n_estimators'] = 300
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['bootstrap'] = True

model = RandomForestRegressor(
    max_depth=h_params["max_depth"], 
    n_estimators=h_params["n_estimators"],
    bootstrap=h_params["bootstrap"],
    max_features=h_params["max_features"],
    max_samples=h_params["max_samples"],
    random_state=random_state, 
    verbose=1,
    n_jobs=n_jobs
)

fitted_model = model.fit(phenos_t, scores_t)

results = permutation_importance(
    fitted_model, phenos_v, scores_v, n_repeats=num_repeats,
    random_state=0, scoring='neg_mean_absolute_error', n_jobs=n_jobs,
    random_state = random_state
)

results_stats = [] 

for i in results.importances_mean.argsort()[::-1]:
    results_stats.append([phenos_subset[i], results.importances_mean[i], results.importances_std[i]])

results_stats = pd.DataFrame(results_stats, columns = ['feature', 'importance_mean', 'importance_std'])

results_stats = results_stats.sort_values(by='importance_mean', ascending=False)
filename = 'Analysis/RandomForest/feature_importance_perm_rankings_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())

results_stats.to_csv(filename)

print('Complete.')