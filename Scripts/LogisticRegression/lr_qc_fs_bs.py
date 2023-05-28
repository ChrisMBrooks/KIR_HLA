# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

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
n_repeats = 5
random_state_1 = 42
random_state_2 = 21

impute = True
standardise = False
normalise = False
strategy='mean'

forward_selection = True
backward_selection = True
tolerance = None
n_features_to_select='auto'
scoring_method='neg_mean_absolute_error'

source_filename = 'Analysis/LogisticRegression/26042023/{}/lr_gs_feature_weights_{}_26042023.csv'
params_filename = 'Analysis/LogisticRegression/lr_gs_candidate_h_params_26042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
output_filename = "Analysis/LogisticRegression/lr_{}_candidate_features_{}_{}.csv"

#Import Config Params
h_params = pd.read_csv(params_filename, index_col=0)

#Import Data
scores_df = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_df = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')


ikir_labels = ['kir2dl1', 'kir2dl2_s', 'kir2dl2_w', 'kir2dl3', 'kir3dl1']
for key in ikir_labels:
    #Retreive Subset 
    kir_tag = 'f_{}'.format(key)
    phenos_subset = pd.read_csv(source_filename.format(key.upper(), kir_tag), index_col=0)
    phenos_subset = list(phenos_subset['label'].values)
    phenos = phenos_df[phenos_subset].copy()
    # Standardise Data
    scores = scores_df[key].values.reshape(-1,1)
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
    l1_ratio = float(h_params[h_params['label'] == key]['l1_ratio'].values[0])
    C = float(h_params[h_params['label'] == key]['C'].values[0])
    
    # Instantiate Model    
    model = LogisticRegression(
        solver = 'saga',
        penalty = 'elasticnet', 
        l1_ratio = l1_ratio, 
        C = C,
        fit_intercept = True,
        max_iter = 100, 
        n_jobs = n_jobs
    )

    if forward_selection:
        print('Starting Forward Selection...')
        sfs_for = SequentialFeatureSelector(
            model, direction='forward', 
            n_features_to_select=n_features_to_select, 
            scoring=scoring_method, 
            tol=tolerance, cv=cv, n_jobs=n_jobs
        )

        sfs_for.fit(phenos, scores)
        for_selected_features = sfs_for.get_support()
        print('Forward Selection Complete.')

    if backward_selection:
        print('Starting Backward Selection...')
        sfs_bac = SequentialFeatureSelector(
            model, direction='backward', 
            n_features_to_select=n_features_to_select, 
            scoring=scoring_method, 
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

    output_filename.format(flag, key, date_str)
    summary_df.to_csv(output_filename)
print('Complete.')

