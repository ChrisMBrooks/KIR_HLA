import os, random, math, time, uuid
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from Controllers.DataScienceManager import DataScienceManager as dsm

def get_partitioned_data(phenos_subset:str):
    phenos_t, scores_t = get_data('training', phenos_subset)
    phenos_v, scores_v = get_data('validation', phenos_subset)

    return phenos_t, scores_t, phenos_v, scores_v

def get_data(partition:str, phenos_subset):
    scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition=partition)
    phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition=partition)
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
#cut_off = 10
#filename = "Data/Subsets/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(cut_off))


filename = 'Analysis/RandomForest/feature_importance_impurity_rankings_13042023.csv'
phenos_subset = pd.read_csv(filename, index_col=0)
#phenos_subset = phenos_subset[phenos_subset > cut_off].dropna()
phenos_subset = list(phenos_subset.index)[:100]

phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset)

# Instantiate Model & Hyper Params
h_params = dict()

h_params['max_depth'] = 200
h_params['n_estimators'] = 300
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['bootstrap'] = True

# Instantiate Model    
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
# Note if bootstrap is True, max_samples is the number of samples to draw from 
# X to train each base estimator.

model.fit(phenos_t, scores_t)

feature_weights = pd.Series(model.feature_importances_, index=phenos_subset)
feature_weights = feature_weights.sort_values(ascending=False)
#filename = 'Analysis/RandomForest/feature_importance_impurity_rankings_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())
#feature_weights.to_csv(filename)

# Computer Predictions and Summary Stats
y_hat = model.predict(phenos_v)
neg_mae = -1*mean_absolute_error(scores_v, y_hat)

print('best fit mae:', neg_mae)
print('max_depth:', h_params["max_depth"])
print('n_estimators:', h_params["n_estimators"])
print('max_features:', h_params["max_features"])
print('max_samples:', h_params["max_samples"])

run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py