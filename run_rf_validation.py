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

def get_partitioned_data(phenos_subset:str, partition_training_dataset:bool, n_splits:int, random_state:int):
    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos_t = phenos_t[phenos_subset].copy()

    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
    phenos_v = phenos_v[phenos_subset].copy()

    if partition_training_dataset:
        phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
            phenos_t, scores_t, n_splits, random_state
        )

    return phenos_t, scores_t, phenos_v, scores_v

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        impute, strategy, standardise, normalise 
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v)

    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.preprocess_data_v(
        X_t=phenos_t, Y_t=scores_t, X_v=phenos_v, Y_v=scores_v,
        impute = impute, strategy=strategy, standardise = standardise, 
        normalise = normalise
    )

    scores_t = scores_t.ravel()
    scores_v = scores_v.ravel()
    return phenos_t, scores_t, phenos_v, scores_v

def get_final_score(phenos_subset, h_params:dict, 
        validation_approach:str, n_splits:int, random_state:int,
        impute:bool, strategy:str, standardise:bool, normalise:bool
    ):

    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(
        phenos_subset, False, n_splits, random_state)
    
    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, 
        impute, strategy, standardise, normalise
    )

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

    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=h_params["bootstrap"],
        max_features=h_params["max_features"],
        max_samples=h_params["max_samples"],
        min_samples_split = h_params['min_samples_split'],
        random_state=42, 
        verbose=0,
        n_jobs=-1
    )

    model.fit(phenos_t, scores_t)
    
    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos_v)
    neg_mae = -1*mean_absolute_error(scores_v, y_hat)

    return neg_mae

def get_baseline(phenos_subset, h_params, random_state, n_jobs, impute, strategy, standardise, normalise):
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset, False, n_splits, random_state)

    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, 
        impute, strategy, standardise, normalise
    )

    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=h_params["bootstrap"],
        max_features=h_params["max_features"],
        max_samples=h_params["max_samples"],
        random_state=random_state, 
        verbose=0,
        n_jobs=n_jobs
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

# Declare Config Params 
n_jobs = 16 - 1
n_splits = 4
random_state = 42

impute = True
strategy = 'mean'
standardise = True
normalise = True

h_params = dict()
h_params['max_depth'] = 6
h_params['n_estimators'] = 300
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['bootstrap'] = True
h_params['min_samples_split'] = 40

source_filename = 'Analysis/RandomForest/20042023_c0.95_100/r_forest_fs_bs_candidate_features_100_20042023_3.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
output_filename = 'Analysis/RandomForest/rf_final_score_{}.csv'.format(date_str)

# Pull Data from DB
phenos_subset = pd.read_csv(source_filename, index_col=0)
indeces = phenos_subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

validation_approaches = ['tt', 'vv', 'tv']
output = {}

output['baseline'] = get_baseline(
    phenos_subset, h_params, random_state, n_jobs, 
    impute, strategy, standardise, normalise
)

for idx, approach in enumerate(validation_approaches):
    neg_mae = get_final_score(
        phenos_subset=phenos_subset,
        h_params=h_params, 
        validation_approach=approach,
        n_splits=n_splits, 
        random_state = random_state,
        impute = impute, 
        strategy = strategy, 
        standardise = standardise, 
        normalise = normalise
    )
    key = 'neg_mae' + '_' + validation_approaches[idx]
    output[key] = neg_mae

#Export Results
output['features'] = phenos_subset
output = pd.Series(output)
print(output)
output.to_csv(output_filename)

print('Complete.')