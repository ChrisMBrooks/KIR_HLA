# Build and evaluate an Elastic Net Model on a subset of phenos

import pandas as pd
import numpy as np 
import uuid, time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

from Controllers.DataScienceManager import DataScienceManager as dsm

def preprocess_data(X, Y):
    X, Y = impute_missing_values(X, Y, strategy='mean')
    X, Y = standardise(X, Y)
    X, Y = normalise(X, Y, min=0, max=1) #Centeres about the origin
    return X, Y

def impute_missing_values(X, Y, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    X = imputer.fit_transform(X)
    Y = imputer.fit_transform(Y)
    return X, Y

def standardise(X,Y):
    x_std_sclr = StandardScaler()
    y_std_sclr = StandardScaler()

    X = x_std_sclr.fit_transform(X)
    Y = y_std_sclr.fit_transform(Y)
    return X, Y

def normalise(X, Y, min, max):
    x_mms = MinMaxScaler(feature_range=(min, max))
    y_mms = MinMaxScaler(feature_range=(min, max))

    X = x_mms.fit_transform(X)

    # MMS scales and translates each feature individually
    Y = y_mms.fit_transform(Y) 

    return X, Y

def retrieve_subset_of_data(filename:str, partition = 'training'):
    # Get iKIR Scores
    ikir_scores = dsci_mgr.data_mgr.feature_values(normalise = False, fill_na = False, 
        fill_na_value = 0.0, partition = partition
    )

    # Read in List of Desired Phenos
    phenos_subset = list(pd.read_csv(filename, index_col=0).values[:,0])
    outcomes_ss_df = dsci_mgr.data_mgr.outcomes_subset(desired_columns=phenos_subset, partition='training')
    
    # Filter out Phenos with too many NaNs
    columns = outcomes_ss_df.columns[1:-2]
    nans_sums_df  = outcomes_ss_df[columns].isna().sum()
    ss_labels = list(nans_sums_df[nans_sums_df<10].axes[0])

    # Filter Master Dataset on Desired Subset
    immunos_maxtrix_subset = dsci_mgr.data_mgr.outcomes_subset(desired_columns=ss_labels, 
        partition = partition
    )
        
    immunos_labels = immunos_maxtrix_subset.columns[1:-2]
    immunos_maxtrix_subset = immunos_maxtrix_subset.values[:,1:-2].astype(float)
    return ikir_scores, immunos_labels, immunos_maxtrix_subset

# Begin Script
start_time = time.time()
use_full_dataset = True
dsci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

filename = "Data/unlike_phenos_27022023.csv"
ikir_scores_t_raw, immunos_labels_t, immunos_maxtrix_subset_t = \
    retrieve_subset_of_data(filename, partition = 'training')

# Preprocess Data (Impute, Standardise, Normalise)
ikir_scores_t, immunos_maxtrix_subset_t = preprocess_data(ikir_scores_t_raw, immunos_maxtrix_subset_t)

run_id = str(uuid.uuid4().hex)

n_splits = 2
n_repeats = 2
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

# Define Hyper Params Grid
hyper_params_grid = dict()
l1_ratio_min = 0.1
l1_ratio_max = 1.0
l1_ratio_step = 0.1
hyper_params_grid['l1_ratio'] = np.arange(
        l1_ratio_min, l1_ratio_max, l1_ratio_step
)

#Alpha = 0 is equivalent to an ordinary least square 
alpha_min = 0.1
alpha_max = 1.0
alpha_step = 0.1
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

grid_search_results = grid_searcher.fit(immunos_maxtrix_subset_t, ikir_scores_t)
best_fit_mae = grid_search_results.best_score_
best_fit_hyper_params = grid_search_results.best_params_
l1_ratio =  best_fit_hyper_params['l1_ratio']
alpha = best_fit_hyper_params['alpha']

model = ElasticNet(alpha=alpha, 
            l1_ratio=l1_ratio
)

model.fit(immunos_maxtrix_subset_t, ikir_scores_t)

feature_weights = model.coef_
non_zero_coeff_count  = np.where(np.absolute(feature_weights) > 0)[0].size

weights_df =pd.DataFrame(zip(immunos_labels_t, feature_weights), columns=['label', 'weight'])

candidates = weights_df[weights_df['weight'] != 0]
print(candidates)

run_time = time.time() - start_time

run_record = [run_id, run_time, float(best_fit_mae), non_zero_coeff_count, float(l1_ratio), l1_ratio_min, 
        l1_ratio_max, l1_ratio_step, float(alpha), alpha_min, alpha_max, alpha_step
]

print(run_record)
print("Complete")
