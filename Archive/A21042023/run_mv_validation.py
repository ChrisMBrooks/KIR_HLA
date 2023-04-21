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

def get_partitioned_data(phenos_subset:str):
    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')

    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')

    scores_tv = pd.concat([scores_t, scores_v])
    phenos_tv = pd.concat([phenos_t, phenos_v])

    phenos_tv, scores_tv = prep_data(phenos_tv, scores_tv, phenos_subset)

    phenos_t = phenos_tv[0:phenos_t.shape[0], :]
    phenos_v = phenos_tv[phenos_t.shape[0]:, :]

    scores_t = scores_tv[0:scores_t.shape[0]]
    scores_v = scores_tv[scores_t.shape[0]:]

    return phenos_t, scores_t, phenos_v, scores_v

def prep_data(phenos, scores, phenos_subset):
    phenos = phenos[phenos_subset]

    # Standardise Data
    scores = scores['f_kir_score'].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        phenos, scores, impute=True, standardise=True, 
        normalise=True, strategy='median'
    )

    scores = scores.ravel()
    return phenos, scores

def get_final_score(phenos_subset, validation_approach=str):
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset)

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
        n_splits = 5
        n_repeats = 10
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
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

def get_baseline(phenos_subset):
    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(phenos_subset)
    neg_mae = -1*mean_absolute_error(scores_v, scores_t.mean()*np.ones(scores_v.shape))

    return neg_mae

#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database)

# Pull Data from DB
#Read in Subset of Immunophenotypes
"""date_str = '07042023'
filename = 'Analysis/Multivariate/{}/feature_importance_perm_rankings_{}.csv'.format(
    date_str, date_str
)"""

filename = 'Analysis/Multivariate/11042023_c_rc3/feature_importance_perm_rankings_11042023.csv'

candidates = pd.read_csv(filename, index_col=0)
phenos_subset = list(candidates.values[:, 0])

validation_approaches = ['tt', 'vv', 'tv', 'tv_cv']
output = {}

output['baseline'] = get_baseline(phenos_subset)
for idx, approach in enumerate(validation_approaches):
    neg_mae = get_final_score(
        phenos_subset=phenos_subset, 
        validation_approach=approach
    )
    key = 'neg_mae' + '_' + validation_approaches[idx]
    output[key] = neg_mae

output['features'] = phenos_subset

output = pd.Series(output)
print(output)

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/Multivariate/final_score_{}.csv'.format(date_str)
output.to_csv(filename)

print('Complete.')