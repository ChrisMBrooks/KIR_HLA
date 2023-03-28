import numpy as np
import pandas as pd
import time, uuid

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

from Controllers.DataScienceManager import DataScienceManager as dsm

start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

# Pull Data from DB

#Read in Subset of Immunophenotypes
cut_off = 10
filename = "Data/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(cut_off))
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
    phenos, scores
)

# Instantiate Model & Hyper Params
n_splits = 2
n_repeats = 2
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

hyper_params_grid = dict()

l1_ratio_min = 0.1
l1_ratio_max = 0.3
l1_ratio_step = 0.05
hyper_params_grid['l1_ratio'] = np.arange(
    l1_ratio_min, l1_ratio_max, l1_ratio_step
)

#Alpha = 0 is equivalent to an ordinary least square 
alpha_min = 0.01
alpha_max = 0.07
alpha_step = 0.01
hyper_params_grid['alpha'] = np.arange(
    alpha_min, alpha_max, alpha_step
)

# Instantiate evaluation method model
model = ElasticNet(alpha=hyper_params_grid['alpha'], 
    l1_ratio=hyper_params_grid['l1_ratio']
)

# Instantiate GridSearch Object
grid_searcher = GridSearchCV(model, hyper_params_grid, 
    scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, 
    verbose=2
)

# Perform Grid Search for Hyper Params
grid_search_results = grid_searcher.fit(phenos, scores)
best_fit_mae = grid_search_results.best_score_
best_fit_hyper_params = grid_search_results.best_params_
l1_ratio =  best_fit_hyper_params['l1_ratio']
alpha = best_fit_hyper_params['alpha']

model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

model.fit(phenos, scores)

feature_weights = model.coef_
non_zero_coeff_count  = np.where(np.absolute(feature_weights) > 0)[0].size

run_time = time.time() - start_time

run_record = [run_id, run_time, float(best_fit_mae), non_zero_coeff_count, 
              float(l1_ratio), l1_ratio_min, l1_ratio_max, l1_ratio_step, 
              float(alpha), alpha_min, alpha_max, alpha_step
]
        
table_columns = ['run_id', 'run_time', 'mae', 'non_zero_coeff_count', 
                 'l1_ratio', 'l1_ratio_min', 'l1_ratio_max', 'l1_ratio_step', 
                 'alpha', 'alpha_min', 'alpha_max', 'alpha_step'
]

print(run_record)
print('Complete.')

