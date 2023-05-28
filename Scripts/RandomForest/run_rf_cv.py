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

# Declare Config Params
n_jobs = 16 - 1
n_splits = 5
n_repeats = 4
random_state = 42

impute = True
strategy = 'median'
normalise = True
standardise = True

h_params = dict()
h_params['max_depth'] = 10
h_params['n_estimators'] = 100
h_params['max_features'] = 0.3
h_params['max_samples'] = 0.8
h_params['bootstrap'] = True
h_params['min_samples_split'] = 20

#source_filename = 'Data/unlike_phenos_0.95_05042023.csv'
source_filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()

output_filename = 'Analysis/RandomForest/rf_results_w_cv_{}.csv'.format(date_str)
weights_filename = 'Analysis/RandomForest/rf_feature_importance_impurity_rankings_{}.csv'.format(date_str)

#Read in Subset of Immunophenotypes
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

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

cv = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=random_state
)
splits_gen = cv.split(phenos)

results = []
for i in range(0, n_repeats+1):
    split = next(splits_gen)
    train_indeces = split[0]
    test_indeces = split[1]

    model.fit(phenos[train_indeces, :], scores[train_indeces])

    feature_weights = pd.Series(model.feature_importances_, index=phenos_subset)
    feature_weights = feature_weights.sort_values(ascending=False)
    feature_weights.to_csv(weights_filename)

    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos[test_indeces, :])
    neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
    results.append(neg_mae)

results = np.array(results)
neg_mae = results.mean()

output = {}
output['data_source'] = source_filename
output['avg_neg_mae'] = neg_mae
output['max_depth'] = h_params['max_depth']
output['n_estimators'] = h_params['n_estimators']
output['max_features'] = h_params['max_features']
output['max_samples'] = h_params['max_samples']
output['bootstrap'] = h_params['bootstrap']
output['min_samples_split'] = h_params['min_samples_split']

output['n_splits'] = n_splits
output['n_repeats'] = n_repeats
output['random_state'] = random_state

output['impute'] = impute
output['standardise'] = standardise
output['normalise'] = normalise
output['strategy='] = strategy

output['run_id'] = run_id

output = pd.Series(output)
output.to_csv(output_filename)

print(output)

run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py