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

# Declare Config Parameters
partition_training_dataset = True
n_jobs = 16 - 1
random_state = 42
n_splits = 4
impute = True
strategy='mean'
standardise = True
normalise = True 

h_params = dict()
h_params['max_depth'] = 200
h_params['n_estimators'] = 300
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['bootstrap'] = True
h_params['min_samples_split'] = 40

source_filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()

weights_filename = 'Analysis/RandomForest/rf_feature_importance_impurity_rankings_{}.csv'.format(date_str)
summary_filename = 'Analysis/RandomForest/rf_results_{}.csv'.format(date_str)


#Pull Data from DB
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_t = phenos_t[phenos_subset]

scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos_v = phenos_v[phenos_subset]

# Partition Data
if partition_training_dataset:
    #Overwites Validation Data If Required.
    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
        phenos_t, scores_t, n_splits, random_state
)

# Massage Data
phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
    phenos_t, scores_t, phenos_v, scores_v, impute, strategy, 
    standardise, normalise
)

# Instantiate Model    
model = RandomForestRegressor(
    max_depth=h_params["max_depth"], 
    n_estimators=h_params["n_estimators"],
    bootstrap=h_params["bootstrap"],
    max_features=h_params["max_features"],
    max_samples=h_params["max_samples"],
    min_samples_split=h_params['min_samples_split'],
    random_state=random_state, 
    verbose=0,
    n_jobs=n_jobs
)
# Note if bootstrap is True, max_samples is the number of samples to draw from 
# X to train each base estimator.

model.fit(phenos_t, scores_t)

feature_weights = pd.Series(model.feature_importances_, index=phenos_subset)
feature_weights = feature_weights.sort_values(ascending=False)

feature_weights.to_csv(weights_filename)

# Computer Predictions and Summary Stats
y_hat = model.predict(phenos_v)
neg_mae = -1*mean_absolute_error(scores_v, y_hat)

output = {}
output['data_source'] = source_filename
output['overall_neg_mae'] = neg_mae
output['max_depth'] = h_params['max_depth']
output['n_estimators'] = h_params['n_estimators']
output['max_features'] = h_params['max_features']
output['max_samples'] = h_params['max_samples']
output['bootstrap'] = h_params['bootstrap']
output['min_samples_split'] = h_params['min_samples_split']
output['run_id'] = run_id

output = pd.Series(output)
output.to_csv(summary_filename)

print(output)

run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py