import numpy as np
import pandas as pd
import time, uuid

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from Controllers.DataScienceManager import DataScienceManager as dsm

def get_partitioned_data(phenos_subset:str, partition_training_dataset:bool, n_splits:int, random_state:int):
    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos_t = phenos_t[phenos_subset].copy()

    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
    phenos_v = phenos_v[phenos_subset].copy()

    if partition_training_dataset:
        phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
            phenos_t, scores_t, n_splits, random_state
        )

    return phenos_t, scores_t, phenos_v, scores_v

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        impute, strategy, standardise, normalise 
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v)

    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.preprocess_data_v(
        X_t=phenos_t, Y_t=scores_t, X_v=phenos_v, Y_v=scores_v,
        impute = impute, strategy=strategy, standardise = standardise, 
        normalise = normalise
    )

    scores_t = scores_t.ravel()
    scores_v = scores_v.ravel()
    return phenos_t, scores_t, phenos_v, scores_v

def get_final_score(phenos_subset, validation_approach:str, 
                    n_splits, random_state, 
                    impute, strategy, standardise, normalise
    ):
    
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset, False, n_splits, random_state)

    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, 
        impute, strategy, standardise, normalise
    )

    cross_val = False
    if validation_approach == 'tt':
        phenos_t = phenos_t
        phenos_v = phenos_t
        scores_t = scores_t
        scores_v = scores_t
    elif validation_approach == 'tv':
        phenos_t = phenos_t
        phenos_v = phenos_v
        scores_t = scores_t
        scores_v = scores_v
    elif validation_approach == 'tv_cv':
        phenos_t = phenos_t
        phenos_v = phenos_v
        scores_t = scores_t
        scores_v = scores_v
        cross_val = True
    elif validation_approach == 'vv':
        phenos_t = phenos_v
        phenos_v = phenos_v
        scores_t = scores_v
        scores_v = scores_v


    # Fit the Multivar Linear Regression Model
    if cross_val:
        cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=random_state)
        model = LinearRegression()
        cv_results = cross_validate(
            model, phenos_t, scores_t, cv=cv, 
            return_estimator=True, scoring='neg_mean_absolute_error'
        )

        idx = np.argmax(cv_results['test_score'])
        optimal_socre = cv_results['test_score'][idx]

        coef_ = cv_results['estimator'][idx].coef_
        intercept_ = cv_results['estimator'][idx].intercept_

        model = LinearRegression()
        model.coef_ = coef_
        model.intercept_ = intercept_
    else:
        model = LinearRegression()
        model.fit(phenos_t, scores_t)
    
    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos_v)
    neg_mae = -1*mean_absolute_error(scores_v, y_hat)

    return neg_mae

def get_baseline(phenos_subset, n_splits, random_state, impute, strategy, standardise, normalise):
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset, False, n_splits, random_state)

    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, 
        impute, strategy, standardise, normalise
    )

    neg_mae = -1*mean_absolute_error(scores_v, scores_t.mean()*np.ones(scores_v.shape))

    return neg_mae

#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database)

#Declare Config Params

impute = True
strategy = 'mean'
standardise = True
normalise = True

random_state = 42
n_splits = 4

source_filename = 'Analysis/Multivariate/11042023_c_rc3/feature_importance_perm_rankings_11042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
output_filename = 'Analysis/Multivariate/final_score_{}.csv'.format(date_str)

# Pull Data from DB
#Read in Subset of Immunophenotypes
candidates = pd.read_csv(source_filename, index_col=0)
phenos_subset = list(candidates.values[:, 0])

# Evaluate Models
validation_approaches = ['tt', 'vv', 'tv', 'tv_cv']
output = {}

output['baseline'] = get_baseline(phenos_subset, n_splits, random_state, 
    impute, strategy, standardise, normalise
)
for idx, approach in enumerate(validation_approaches):
    neg_mae = get_final_score(
        phenos_subset = phenos_subset, 
        validation_approach = approach,
        n_splits = n_splits, 
        random_state = random_state, 
        impute = impute, 
        strategy = strategy, 
        standardise = standardise, 
        normalise = normalise
    )
    key = 'neg_mae' + '_' + validation_approaches[idx]
    output[key] = neg_mae

# Export Results
output['features'] = phenos_subset
output = pd.Series(output)
output.to_csv(output_filename)
print(output)

print('Complete.')