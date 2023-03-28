# Experimenting with dataset filtering by removing immunophenotypes 
# with greater than a defined number of NaNs

import time, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

#Read in Subset of Immunophenotypes
filter = 20
filename = "Data/14022023/na_filtered_phenos_less_thn_{}_zeros_14022023.csv".format(str(filter))
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

#Filter Dataset on Desired Subset 
immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset)
#immunos_maxtrix_subset.fillna(0.0, inplace=True)
immunos_maxtrix_subset = immunos_maxtrix_subset.values[:,1:-2]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Y = imputer.fit_transform(immunos_maxtrix_subset)

X = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
    partition = 'training')

# Standardise Data
x_std_sclr = StandardScaler()
y_std_sclr = StandardScaler()

X = x_std_sclr.fit_transform(X)
Y = y_std_sclr.fit_transform(Y)

# Normalise Data Around the Origin
x_mms = MinMaxScaler(feature_range=(-1, 1))
y_mms = MinMaxScaler(feature_range=(-1, 1))

X = x_mms.fit_transform(X)
Y = y_mms.fit_transform(Y) # MMS scales and translates each feature individually

#Sigmoid Transformation
#Y = 1/(1+np.exp(-1*Y))

# Fit the Model 
start_time = time.time()
run_id = str(uuid.uuid4().hex)

n_splits = 2
n_repeats = 3
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

hyper_params_grid = dict()

l1_ratio_min = 0.1
l1_ratio_max = 1.1
l1_ratio_step = 0.1
hyper_params_grid['l1_ratio'] = np.arange(
    l1_ratio_min, l1_ratio_max, l1_ratio_step
)

#Alpha = 0 is equivalent to an ordinary least square 

# alpha_min = 1.1
# alpha_max = 1.2
# alpha_step = 1.3

alpha_min = 0.15
alpha_max = 0.5
alpha_step = 0.05
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

subset_note = "<{} NaNs".format(filter)

run_record = [run_id, run_time, float(best_fit_mae), non_zero_coeff_count, float(l1_ratio), l1_ratio_min, 
    l1_ratio_max, l1_ratio_step, float(alpha), alpha_min, alpha_max, alpha_step, subset_note
]
        
table_columns = ['run_id', 'run_time', 'mae', 'non_zero_coeff_count', 'l1_ratio', 
    'l1_ratio_min', 'l1_ratio_max', 'l1_ratio_step', 'alpha', 
    'alpha_min', 'alpha_max', 'alpha_step', 'subset'
]

# sql.insert_records(schema_name='KIR_HLA_STUDY', table_name='model_result_el_net_subset',
#     column_names=table_columns, values=[run_record]
# )

print(run_record)
print('Complete.')

