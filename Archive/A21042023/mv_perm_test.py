# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, random, math
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
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
        X=phenos, Y=scores, impute = True, standardise = True, 
        normalise = True, strategy='median'
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

#Read in Subset of Immunophenotypes
segregate_dataset = True
num_repeats = 10
n_jobs = 16 - 1
random_state = 42

"""
cut_off = 10
filename = "Data/Subsets/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(cut_off))
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])
"""
"""date_str = '08042023'
filename = 'Analysis/Multivariate/multivar_qc_fs_bs_candidate_features_{}.csv'.format(date_str) """
filename = 'Analysis/Multivariate/11042023_c_rc3/multivar_qc_fs_bs_candidate_features_11042023.csv'
phenos_subset = pd.read_csv(filename, index_col=0)
indeces = phenos_subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

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

model = LinearRegression()

fitted_model = model.fit(phenos_t, scores_t)

results = permutation_importance(fitted_model, phenos_v, scores_v, n_repeats=num_repeats,
    random_state=random_state, scoring ='neg_mean_absolute_error', n_jobs=n_jobs
)

results_stats = [] 

for i in results.importances_mean.argsort()[::-1]:
    results_stats.append([phenos_subset[i], results.importances_mean[i], results.importances_std[i]])

results_stats = pd.DataFrame(results_stats, columns = ['feature', 'importance_mean', 'importance_std'])

results_stats = results_stats.sort_values(by='importance_mean', ascending=False)
filename = 'Analysis/Multivariate/feature_importance_perm_rankings_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())

results_stats.to_csv(filename)

print('Complete.')

"""

The permutation feature importance is the decrease in a model score when a single feature value is randomly shuffled.

Permutation importances can be computed either on the training set or on a held-out testing or validation set. 
Using a held-out set makes it possible to highlight which features contribute the most to 
the generalization power of the inspected model.

"""