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
    print('Setting Sys Params:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    count = int(sys.argv[1])*25
    forward_selection = bool(sys.argv[2])
    backward_selection = bool(sys.argv[3])
    test_id = int(sys.argv[4])
else:
    count = 100
    forward_selection = True
    backward_selection = True
    test_id = 1

use_full_dataset=True
use_database = False

#Instantiate Controllers
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

# Declare Config Params
n_jobs = 24 - 1
n_splits = 4
n_repeats = 10
random_state_1 = 84
random_state_2 = 168
tolerance = None

impute = True
standardise = True
normalise = True
strategy='median'

h_params = dict()

h_params['max_depth'] = 6
h_params['n_estimators'] = 100
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['min_samples_split'] = [40]
h_params['bootstrap'] = True

#Read in Subset of Immunophenotypes
source_filename = "Analysis/RandomForest/April/20042023/rf_feature_importance_impurity_rankings_20042023_{}.csv".format(test_id)
phenos_subset = pd.read_csv(source_filename, index_col=0)
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
results_filename = "Analysis/RandomForest/rf_{}_candidate_features_{}_{}_{}.csv".format(flag, count, date_str, test_id)
summary_df.to_csv(results_filename)

run_details_filename = "Analysis/RandomForest/rf_{}_run_details_{}_{}_{}.csv".format(flag, count, date_str, test_id)
output = dict()
output['data_source'] = source_filename
output['test_id'] = test_id
output['selection_type'] = flag
output['cut_off_threshold'] = count

output['n_splits'] = n_splits
output['n_repeats'] = n_repeats
output['random_state_1'] = random_state_1
output['random_state_2'] = random_state_2
output['tolerance'] = tolerance

output['impute'] = impute
output['standardise'] = standardise
output['normalise'] = normalise
output['strategy='] = strategy

for key in h_params:
    output[key] = h_params[key]

output = pd.Series(output)
output.to_csv(run_details_filename)
print('Complete.')
