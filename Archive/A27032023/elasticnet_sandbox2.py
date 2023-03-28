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
from sklearn.preprocessing import MinMaxScaler
import uuid

# Meta Calculations
start_time = time.time()
run_id = str(uuid.uuid4().hex)

# Normalise Data
x_mms = MinMaxScaler(feature_range=(-1, 1))
y_mms = MinMaxScaler(feature_range=(-1, 1))

X = x_mms.fit_transform(X)
Y = y_mms.fit_transform(Y) # MMS scales and translates each feature individually

Y = 1/(1+np.exp(-1*Y))

#Instantiate Cross Fold Object
n_splits = 2
n_repeats = 2
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

#Instantiate Hyper Params
hyper_params_grid = dict()
l1_ratio_min = 0.1
l1_ratio_max = 0.95
l1_ratio_step = 0.1
hyper_params_grid['l1_ratio'] = np.arange(
    l1_ratio_min, l1_ratio_max, l1_ratio_step
)

#Alpha = 0 is equivalent to an ordinary least square 
alpha_min = 0.1
alpha_max = 1.5
alpha_step = 0.1
hyper_params_grid['alpha'] = np.arange(
    alpha_min, alpha_max, alpha_step
)

#Instantiate Model
model = ElasticNet(alpha=hyper_params_grid['alpha'], 
    l1_ratio=hyper_params_grid['l1_ratio']
)

#Instantiate Gride Search Obj
grid_searcher = GridSearchCV(model, hyper_params_grid, 
    scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, 
    verbose=2
)

#Perform Grid Search
grid_search_results = grid_searcher.fit(Y, X)
best_fit_mae = grid_search_results.best_score_
best_fit_hyper_params = grid_search_results.best_params_
l1_ratio =  best_fit_hyper_params['l1_ratio']
alpha = best_fit_hyper_params['alpha']

#Fit Final Model
model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

model.fit(Y, X)

#Report Model Results to DB
feature_weights = model.coef_

non_zero_coeffs = np.where(np.absolute(feature_weights) > 0)[0]

non_zero_coeff_count = non_zero_coeffs.size

pheno_labels = data_mgr.outcomes(fill_na = False, partition = 'training').columns[1:-2]
for index, item in enumerate(feature_weights):
    if np.absolute(item):
        print(pheno_labels[index], item)

run_time = time.time() - start_time

run_record = [run_id, run_time, float(best_fit_mae), non_zero_coeff_count, float(l1_ratio), l1_ratio_min, 
    l1_ratio_max, l1_ratio_step, float(alpha), alpha_min, alpha_max, alpha_step
]
        
table_columns = ['run_id', 'run_time', 'mae', 'non_zero_coeff_count', 'l1_ratio', 
    'l1_ratio_min', 'l1_ratio_max', 'l1_ratio_step', 'alpha', 
    'alpha_min', 'alpha_max', 'alpha_step'
]

# sql.insert_records(schema_name='KIR_HLA_STUDY', table_name='model_result_el_net_norm',
#         column_names=table_columns, values=[run_record]
# )

print(run_record)
print('Complete.')

