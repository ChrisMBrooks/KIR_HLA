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
filename0 = 'Data/Subsets/clustered_0.8_and_restricted_to_phenos_with_less_thn_10_zeros_20042023.csv'
phenos_subset = list(pd.read_csv(filename0, index_col=0).values[:, 0])

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
    X=phenos, Y=scores, impute = True, standardise = True, 
    normalise = True, strategy='mean'
)

scores = scores.ravel()

# Instantiate Model & Hyper Params
h_params = dict()

h_params['max_depth'] = 200
h_params['n_estimators'] = 300
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.9
h_params['bootstrap'] = True
h_params['min_samples_split'] = 40

# Instantiate Model    
model = RandomForestRegressor(
    max_depth=h_params["max_depth"], 
    n_estimators=h_params["n_estimators"],
    bootstrap=h_params["bootstrap"],
    max_features=h_params["max_features"],
    max_samples=h_params["max_samples"],
    min_samples_split=h_params['min_samples_split'],
    random_state=42, 
    verbose=1,
    n_jobs=-1
)
# Note if bootstrap is True, max_samples is the number of samples to draw from 
# X to train each base estimator.

cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
splits_gen = cv.split(phenos)
split = next(splits_gen)
train_indeces = split[0]
test_indeces = split[1]

model.fit(phenos[train_indeces, :], scores[train_indeces])

feature_weights = pd.Series(model.feature_importances_, index=phenos_subset)
feature_weights = feature_weights.sort_values(ascending=False)
filename = 'Analysis/RandomForest/feature_importance_impurity_rankings_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())
feature_weights.to_csv(filename)

# Computer Predictions and Summary Stats
y_hat = model.predict(phenos[test_indeces, :])
neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)

output = {}
output['data_source'] = filename0
output['overall_neg_mae'] = neg_mae
output['max_depth'] = h_params['max_depth']
output['n_estimators'] = h_params['n_estimators']
output['max_features'] = h_params['max_features']
output['max_samples'] = h_params['max_samples']
output['bootstrap'] = h_params['bootstrap']
output['min_samples_split'] = h_params['min_samples_split']
output['run_id'] = run_id

output = pd.Series(output)
filename = 'Analysis/RandomForest/rf_results_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())
output.to_csv(filename)

print(output)

run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py