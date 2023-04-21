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
filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

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
    X=phenos, Y=scores, impute = True, standardise = False, 
    normalise = True
)

scores = scores.ravel()

# Instantiate Model & Hyper Params
n_splits = 4
n_repeats = 5
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

hyper_params_grid = dict()

hyper_params_grid['max_depth'] = [200]
hyper_params_grid['n_estimators'] = [300]
hyper_params_grid['max_features'] = [0.3]
hyper_params_grid['max_samples'] = [0.9]
hyper_params_grid['bootstrap'] = [True]

max_nodes_step = 5
max_nodes_min = 5
max_nodes_max = 50 + max_nodes_step

# max_depth: the maximum depth of the tree 
hyper_params_grid['max_depth'] = np.arange(
    max_nodes_min, max_nodes_max, max_nodes_step
)

# n_estimators: the number of trees in the forest
num_trees_step = 20
num_trees_min = 20
num_trees_max = 300 + num_trees_step

hyper_params_grid['n_estimators'] = np.arange(
    num_trees_min, num_trees_max, num_trees_step
)

gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=hyper_params_grid,
    cv=cv, 
    scoring='neg_mean_absolute_error', 
    verbose=1, 
    n_jobs=-1 # parallelism, -1 means using all processors
)

# Perform Grid Search
grid_search_results = gsc.fit(phenos, scores) # Regress Immunos on iKIR Score
h_params = grid_search_results.best_params_

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

model.fit(phenos, scores)

feature_weights = pd.Series(model.feature_importances_, index=phenos_subset)
feature_weights = feature_weights.sort_values(ascending=False)
filename = 'Analysis/RandomForest/feature_importance_impurity_rankings_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())
feature_weights.to_csv(filename)

# Computer Predictions and Summary Stats
y_hat = model.predict(phenos)
neg_mae = -1*mean_absolute_error(scores, y_hat)

print('best fit mae:', neg_mae)
print('max_depth:', h_params["max_depth"])
print('n_estimators:', h_params["n_estimators"])
print('max_features:', h_params["max_features"])
print('max_samples:', h_params["max_samples"])

#Plot the Grid Search
h_params_sets = grid_search_results.cv_results_['params']
neg_mae_scores = grid_search_results.cv_results_['mean_test_score']

grid_results = [[h_params_sets[i]['max_depth'], h_params_sets[i]['n_estimators'], neg_mae_scores[i]]  \
           for i in range(0, len(h_params_sets),1)]

grid_results = pd.DataFrame(grid_results, columns=['max_depth', 'n_estimators', 'neg_mae'])

sns.relplot(
    data=grid_results, x="max_depth", y="n_estimators", hue="neg_mae", palette='YlOrBr',
)

filename = 'Analysis/RandomForest/grid_search_heat_map_{}.png'.format(data_sci_mgr.data_mgr.get_date_str())
plt.savefig(filename)
plt.show()

run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py