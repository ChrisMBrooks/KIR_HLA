import os, random, math, time, uuid, sys
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

#Set Script Config
if len(sys.argv) == 3:
    start_idx = int(sys.argv[1])
    step = int(sys.argv[2])
    start_idx = start_idx*step
else:
    start_idx = 0
    step = 50

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
strategy = 'mean'
normalise = True
standardise = True

source_filename = 'Data/unlike_phenos_0.95_05042023.csv'
h_params_filename = 'Data/grid_search_parmas_13052023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
output_filename = 'Analysis/RandomForest/rf_parallel_gs_results_{}_{}_{}.csv'.format(start_idx, step, date_str)

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

#Import Matrix of H Params
h_params_df = pd.read_csv(h_params_filename, index_col=0)

# Run Models
records = []
for idx in range(start_idx, start_idx + step, 1):
    if idx < h_params_df.shape[0]:
        record = dict()

        # Instantiate Model    
        model = RandomForestRegressor(
            max_depth=h_params_df.iloc[idx]["max_depth"], 
            n_estimators=h_params_df.iloc[idx]["n_estimators"],
            max_features=h_params_df.iloc[idx]["max_features"],
            max_samples=h_params_df.iloc[idx]["max_samples"],
            bootstrap=h_params_df.iloc[idx]["bootstrap"],
            min_samples_split=h_params_df.iloc[idx]['min_samples_split'],
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

            # Computer Predictions and Summary Stats
            y_hat = model.predict(phenos[test_indeces, :])
            neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
            results.append(neg_mae)

        results = np.array(results)
        neg_mae = results.mean()

        record['index'] = idx
        record['max_depth'] = h_params_df.iloc[idx]['max_depth']
        record['n_estimators'] = h_params_df.iloc[idx]['n_estimators']
        record['max_features'] = h_params_df.iloc[idx]['max_features']
        record['max_samples'] = h_params_df.iloc[idx]['max_samples']
        record['bootstrap'] = h_params_df.iloc[idx]['bootstrap']
        record['min_samples_split'] = h_params_df.iloc[idx]['min_samples_split']
        record['mean_neg_mae'] = neg_mae
        records.append(record)

output = pd.DataFrame(records)
output.to_csv(output_filename)

run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py