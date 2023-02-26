import os, random, math, time
import pandas as pd
import numpy as np

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
iKIR_score_matrix = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

iKIR_score_matrix = iKIR_score_matrix.ravel()

immunos_matrix = data_mgr.outcome_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')
print(immunos_matrix.shape)

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import uuid

start_time = time.time()
run_id = str(uuid.uuid4().hex)

n_splits = 2
n_repeats = 2
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

max_nodes_min = 3
max_nodes_max = 40
max_nodes_step = 1

num_trees_min = 10
num_trees_max = 1000

gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={
        'max_depth': range(max_nodes_min, max_nodes_max, max_nodes_step),
        'n_estimators': (num_trees_min, 50, 100, num_trees_max),
        # max_depth: the maximum depth of the tree 
        # n_estimators: the number of trees in the forest
    },
    cv=cv, 
    scoring='neg_mean_squared_error', 
    verbose=1, 
    n_jobs=-1 # parallelism, -1 means using all processors
)

# Perform Grid Search
grid_result = gsc.fit(immunos_matrix, iKIR_score_matrix) # Regress Immunos on iKIR Score
h_params = grid_result.best_params_

# Instantiate Model    
model = RandomForestRegressor(
    max_depth=h_params["max_depth"], 
    n_estimators=h_params["n_estimators"],
    bootstrap=True,
    max_samples=0.8,
    random_state=False, 
    verbose=1
)

iKIR_score = cross_val_score(model, immunos_matrix, iKIR_score_matrix, 
    cv=cv, scoring='neg_mean_absolute_error'
)

print(iKIR_score)

#r_sqaured = r2_score(iKIR_score_matrix, iKIR_score_hat)
#best_fit_mae = mean_squared_error(iKIR_score_matrix, iKIR_score_hat)

# get importance
""" 
fi_scores = model.feature_importances_
assay_names = list(pheno_train_df.columns)[1:-2]
feature_scores = [[assay_names[i], fi_scores[i]] for i in range(0, len(assay_names))]
feature_scores_df = pd.DataFrame(feature_scores, columns=['feature_id', 'fi_score'])
feature_scores_df.sort_values(by='fi_score', inplace=True, ascending=False) 
"""
run_time = time.time() - start_time 

""" run_record = [run_id, run_time, float(best_fit_mae), float(h_params["max_depth"]), 
    max_nodes_min, max_nodes_max, max_nodes_step, h_params["n_estimators"], 
    num_trees_min, num_trees_max
]
        
table_columns = ['run_id', 'run_time', 'mae', 'max_nodes', 'max_nodes_min', 
    'max_nodes_max', 'max_nodes_step', 'num_trees', 'num_trees_min', 'num_trees_max'
]

sql.insert_records(schema_name='KIR_HLA_STUDY', table_name='model_result_rand_forest',
        column_names=table_columns, values=[run_record]
)

print('Complete.')
 """

