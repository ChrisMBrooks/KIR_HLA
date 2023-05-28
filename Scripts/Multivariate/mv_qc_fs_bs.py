# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

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
forward_selection = True
backward_selection = True

dependent_var = 'kir_count'
scoring = 'neg_mean_absolute_error'

n_jobs = 16 - 1
n_splits = 4
n_repeats = 10
random_state = 42
tolerance = None

impute = True
standardise = False
normalise = False
strategy='mean'

source_filename = 'Analysis/ElasticNet/03052023_count/en_feature_coefs_03052023_4.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
results_filename = "Analysis/Multivariate/multivar_qc_fs_bs_candidate_features_{}.csv".format(date_str)
output_filename = "Analysis/Multivariate/multivar_qc_fs_bs_summary_{}.csv".format(date_str)

#Load Subset of Immunophenotypes
phenos_subset = list(pd.read_csv(source_filename, index_col=0).index)

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos = phenos[phenos_subset]

# Standardise Data
scores = scores[dependent_var].values.reshape(-1,1)
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
    random_state=random_state
)

model = LinearRegression()

if forward_selection:
    print('Starting Forward Selection...')
    sfs_for = SequentialFeatureSelector(
        model, direction='forward', 
        n_features_to_select='auto', 
        scoring=scoring, 
        tol=tolerance, cv=cv, n_jobs=n_jobs
    )

    sfs_for.fit(phenos, scores)
    for_selected_features = sfs_for.get_support()
    print('Forward Selection Complete.')

if backward_selection:
    print('Starting Backward Selection...')
    sfs_bac = SequentialFeatureSelector(
        model, direction='backward', 
        n_features_to_select='auto', 
        scoring=scoring, 
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

summary_df.to_csv(results_filename)

output = {}
output['data_source'] = source_filename
output['dependent_var'] = dependent_var
output['flag'] = flag
output['tolerance'] = tolerance
output['scoring'] = 'neg_mean_absolute_error'
output['impute'] = impute
output['strategy'] = strategy
output['standardise'] = standardise
output['normalise'] = normalise
output['n_splits'] = n_splits
output['n_repeats'] = n_repeats
output['random_state'] = random_state
output = pd.Series(output)
output.to_csv(output_filename)

print('Complete.')

