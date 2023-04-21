import numpy as np
import pandas as pd
import time, uuid

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from Controllers.DataScienceManager import DataScienceManager as dsm

def get_partitioned_data(phenos_subset:str):
    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')

    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')

    scores_tv = pd.concat([scores_t, scores_v])
    phenos_tv = pd.concat([phenos_t, phenos_v])

    phenos_tv, scores_tv = preprocess(phenos_tv, scores_tv, phenos_subset)

    phenos_t = phenos_tv[0:phenos_t.shape[0], :]
    phenos_v = phenos_tv[phenos_t.shape[0]:, :]

    scores_t = scores_tv[0:scores_t.shape[0]]
    scores_v = scores_tv[scores_t.shape[0]:]

    return phenos_t, scores_t, phenos_v, scores_v

def preprocess(phenos, scores, phenos_subset):
    phenos = phenos[phenos_subset]

    # Standardise Data
    scores = scores['f_kir_score'].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        phenos, scores, impute=True, standardise=True, 
        normalise=True, strategy='mean'
    )

    scores = scores.ravel()
    return phenos, scores

def get_final_score(phenos_subset, validation_approach=str):
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset)

    if validation_approach == 'tt':
        phenos_t = phenos_t
        phenos_v = phenos_t
        scores_t = scores_t
        scores_v = scores_t
    elif validation_approach == 'tv':
        phenos_t = phenos_t
        phenos_v = phenos_v
        scores_t = scores_t
        scores_v = scores_v
    elif validation_approach == 'vv':
        phenos_t = phenos_v
        phenos_v = phenos_v
        scores_t = scores_v
        scores_v = scores_v

        
    h_params = dict()
    h_params['max_depth'] = 6
    h_params['n_estimators'] = 300
    h_params['max_features'] = 0.3
    h_params['max_samples'] = 0.9
    h_params['bootstrap'] = True
    h_params['min_samples_split'] = 40

    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=h_params["bootstrap"],
        max_features=h_params["max_features"],
        max_samples=h_params["max_samples"],
        min_samples_split = h_params['min_samples_split'],
        random_state=42, 
        verbose=1,
        n_jobs=-1
    )

    model.fit(phenos_t, scores_t)
    
    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos_v)
    neg_mae = -1*mean_absolute_error(scores_v, y_hat)

    return neg_mae

def get_baseline(phenos_subset):
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset)

    h_params = dict()
    h_params['max_depth'] = 200
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
        random_state=42, 
        verbose=1,
        n_jobs=-1
    )
    
    predictions = []
    n_repeats = 10

    for i in range(n_repeats):
        np.random.shuffle(phenos_t)
        model.fit(phenos_t, scores_t)
        y_hat = model.predict(phenos_v)
        neg_mae = -1*mean_absolute_error(scores_v, y_hat)
        predictions.append(neg_mae)

    neg_mae = np.array(predictions).mean()

    return neg_mae

#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database)

# Pull Data from DB
#Read in Subset of Immunophenotypes
# date_str = '14042023'
# filename = 'Analysis/RandomForest/{}_100/feature_importance_perm_rankings_{}.csv'.format(
#     date_str, date_str
# )
filename = 'Analysis/RandomForest/14042023_100_alt/feature_importance_perm_rankings_14042023.csv'
importances = pd.read_csv(filename, index_col=0)
importances = importances[importances['importance_mean'] -1.0*importances['importance_std'] > 0].copy()

phenos_subset = list(importances['feature'].values)

validation_approaches = ['tt', 'vv', 'tv']
output = {}

output['baseline'] = get_baseline(phenos_subset)
for idx, approach in enumerate(validation_approaches):
    neg_mae = get_final_score(
        phenos_subset=phenos_subset, 
        validation_approach=approach
    )
    key = 'neg_mae' + '_' + validation_approaches[idx]
    output[key] = neg_mae

output['features'] = phenos_subset

output = pd.Series(output)
print(output)

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/RandomForest/final_score_{}.csv'.format(date_str)
output.to_csv(filename)

print('Complete.')