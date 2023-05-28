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

def get_partitioned_data(phenos_subset:str, 
        partition_training_dataset:bool, 
        n_splits:int, n_repeats:int, 
        ith_repeat:int, random_state:int
    ):

    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos_t = phenos_t[phenos_subset].copy()

    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
    phenos_v = phenos_v[phenos_subset].copy()

    if partition_training_dataset:
        phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
            phenos_t, scores_t, 
            n_splits = n_splits, 
            n_repeats = n_repeats, 
            ith_repeat = ith_repeat, 
            random_state = random_state
        )

    return phenos_t, scores_t, phenos_v, scores_v

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        dependent_var:str,
        impute, strategy, standardise, normalise 
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t, dependent_var = dependent_var)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v, dependent_var = dependent_var)

    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.preprocess_data_v(
        X_t=phenos_t, Y_t=scores_t, 
        X_v=phenos_v, Y_v=scores_v,
        impute = impute, 
        strategy=strategy, 
        standardise = standardise, 
        normalise = normalise
    )

    scores_t = scores_t.ravel()
    scores_v = scores_v.ravel()
    return phenos_t, scores_t, phenos_v, scores_v

def get_final_score(phenos_subset, 
                    partition_training_dataset:bool, 
                    validation_approach:str, 
                    n_splits, n_repeats:int, random_state, 
                    dependent_var:str,
                    impute:bool, 
                    strategy:str, 
                    standardise:bool, 
                    normalise:bool
    ):

    results = []
    for i in range(1, n_repeats+1):
        ith_repeat = i

        phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(
            phenos_subset = phenos_subset, 
            partition_training_dataset = partition_training_dataset, 
            n_splits = n_splits, 
            n_repeats = n_repeats, 
            ith_repeat = ith_repeat, 
            random_state = random_state
        )

        phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
            phenos_t, scores_t, phenos_v, scores_v, 
            dependent_var = dependent_var,
            impute = impute, 
            strategy = strategy, 
            standardise = standardise, 
            normalise = normalise
        )

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

        elif validation_approach == 'vv':
            phenos_t = phenos_v
            phenos_v = phenos_v
            scores_t = scores_v
            scores_v = scores_v


        model = LinearRegression()

        model.fit(phenos_t, scores_t)
        
        # Computer Predictions and Summary Stats
        y_hat = model.predict(phenos_v)
        neg_mae = -1*mean_absolute_error(scores_v, y_hat)
        results.append(neg_mae)

    results = np.array(results)
    return results.mean()

def get_baseline(
        phenos_subset, 
        partition_training_dataset:bool, 
        random_state:int, 
        dependent_var:str,
        impute:bool, 
        strategy:str, 
        standardise:bool, 
        normalise:bool
    ):

    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(
        phenos_subset = phenos_subset, 
        partition_training_dataset = partition_training_dataset,
        n_splits = n_splits, 
        n_repeats = 1, 
        ith_repeat = 1,
        random_state = random_state
    )

    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, 
        dependent_var = dependent_var,
        impute = impute, 
        strategy = strategy, 
        standardise = standardise, 
        normalise = normalise
    )

    model = LinearRegression()

    predictions = []
    num_shuffles = 10

    for i in range(num_shuffles):
        model.fit(phenos_t, scores_t)
        shuffled = np.copy(phenos_v)
        np.random.shuffle(shuffled)
        y_hat = model.predict(shuffled)
        neg_mae = -1*mean_absolute_error(scores_v, y_hat)
        predictions.append(neg_mae)

    neg_mae = np.array(predictions).mean()
    return neg_mae

#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database)

#Declare Config Params

dependent_var = 'kir_count'
scoring = 'neg_mean_absolute_error'
fs_bs_filter = 2

partition_training_dataset = True

impute = True
strategy = 'mean'
standardise = True
normalise = True

random_state = 42
n_splits = 4
n_repeats = 5

source_filename = 'Analysis/Multivariate/04052023_count/multivar_qc_fs_bs_candidate_features_04052023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
output_filename = 'Analysis/Multivariate/mv_final_score_{}.csv'.format(date_str)

# Pull Data from DB
#Read in Subset of Immunophenotypes
phenos_subset = pd.read_csv(source_filename, index_col=0)
indeces = phenos_subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= fs_bs_filter)
phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

# Evaluate Models
validation_approaches = ['tt', 'vv', 'tv']
output = {}

output['baseline'] = get_baseline(
    phenos_subset = phenos_subset, 
    partition_training_dataset = partition_training_dataset,
    random_state = random_state, 
    dependent_var = dependent_var,
    impute = impute, 
    strategy = strategy, 
    standardise = standardise, 
    normalise = normalise
)

for idx, approach in enumerate(validation_approaches):
    neg_mae = get_final_score(
        phenos_subset = phenos_subset, 
        partition_training_dataset = partition_training_dataset,
        validation_approach = approach,
        n_splits = n_splits, 
        n_repeats= n_repeats,
        random_state = random_state, 
        dependent_var= dependent_var,
        impute = impute, 
        strategy = strategy, 
        standardise = standardise, 
        normalise = normalise
    )
    key = 'neg_mae' + '_' + validation_approaches[idx]
    output[key] = neg_mae

# Export Results
output['data_source'] = source_filename
output['dependent_var'] = dependent_var
output['fs_bs_filter'] = fs_bs_filter
output['partition_training_dataset'] = partition_training_dataset
output['scoring'] = scoring
output['impute'] = impute
output['strategy'] = strategy
output['standardise'] = standardise
output['normalise'] = normalise
output['n_splits'] = n_splits
output['n_repeats'] = n_repeats
output['random_state'] = random_state
output['features'] = phenos_subset
output = pd.Series(output)
output.to_csv(output_filename)
print(output)

print('Complete.')