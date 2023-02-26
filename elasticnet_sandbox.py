import numpy as np
import pandas as pd
import time

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB

X = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

Y = data_mgr.outcome_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

# Perform Grid Search for Hyper Params
# Instantiate evaluation method model

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import uuid

start_time = time.time()

run_id = str(uuid.uuid4().hex)

n_splits = 2
n_repeats = 2
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

hyper_params_grid = dict()
#hyper_params_grid['alpha'] = [0.01, 0.1, 1.0, 10.0, 100.0]
#hyper_params_grid['l1_ratio'] = np.arange(0.01, 1, 0.1)

#For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty
#L2, aka weight decay, forces weights toward but not exactly zero. 
#L1, absolute value of the weights, may be reduced to zero. 

l1_ratio_min = 0.75
l1_ratio_max = 0.95
l1_ratio_step = 0.05
hyper_params_grid['l1_ratio'] = np.arange(
        l1_ratio_min, l1_ratio_max, l1_ratio_step
)

#Alpha = 0 is equivalent to an ordinary least square 
alpha_min = 980.0
alpha_max = 1200.0
alpha_step = 40.0
hyper_params_grid['alpha'] = np.arange(
        alpha_min, alpha_max, alpha_step
)

model = ElasticNet(alpha=hyper_params_grid['alpha'], 
            l1_ratio=hyper_params_grid['l1_ratio']
)

grid_searcher = GridSearchCV(model, hyper_params_grid, 
    scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, 
    verbose=2
)

grid_search_results = grid_searcher.fit(Y, X)
best_fit_mae = grid_search_results.best_score_
best_fit_hyper_params = grid_search_results.best_params_
l1_ratio =  best_fit_hyper_params['l1_ratio']
alpha = best_fit_hyper_params['alpha']

model = ElasticNet(alpha=alpha, 
            l1_ratio=l1_ratio
)

model.fit(Y, X)

feature_weights = model.coef_
non_zero_coeff_count  = np.where(np.absolute(feature_weights) > 0)[0].size

run_time = time.time() - start_time

run_record = [run_id, run_time, float(best_fit_mae), non_zero_coeff_count, float(l1_ratio), l1_ratio_min, 
        l1_ratio_max, l1_ratio_step, float(alpha), alpha_min, alpha_max, alpha_step
]
        
table_columns = ['run_id', 'run_time', 'mae', 'non_zero_coeff_count', 'l1_ratio', 
        'l1_ratio_min', 'l1_ratio_max', 'l1_ratio_step', 'alpha', 
        'alpha_min', 'alpha_max', 'alpha_step'
]

sql.insert_records(schema_name='KIR_HLA_STUDY', table_name='model_result_el_net',
        column_names=table_columns, values=[run_record]
)

print('Complete.')

