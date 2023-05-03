import os, random, math, time, uuid
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

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

source_filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
params_filename = 'Analysis/LogisticRegression/lr_gs_candidate_h_params_26042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
output_filename = 'Analysis/LogisticRegression/lr_results_w_cv_{}.csv'.format(date_str)

#Import Config Params
h_params = pd.read_csv(params_filename, index_col=0)

#Read in Subset of Immunophenotypes
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

scores_df = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_df = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
ikir_labels = {'f_kir2dl1':1.0, 'f_kir2dl2_s':1.0, 'f_kir2dl2_w':0.5, 'f_kir2dl3':0.75, 'f_kir3dl1':1.0}

predictions_aggregations = {key:[] for key in ikir_labels}
actuals_aggregations = {key:[] for key in ikir_labels}
for key in ikir_labels:
    # Standardise Data
    phenos = phenos_df[phenos_subset]
    scores = scores_df[key].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, placeholder = data_sci_mgr.data_mgr.preprocess_data(
        X=phenos, Y=scores, impute = impute, standardise = standardise, 
        normalise = normalise, strategy = strategy
    )

    scores = scores.ravel()

    kir_tag = key.replace('f_', '')
    l1_ratio = float(h_params[h_params['label'] == kir_tag]['l1_ratio'].values[0])
    C = float(h_params[h_params['label'] == kir_tag]['C'].values[0])
    # Instantiate Model    
    model = LogisticRegression(
        solver = 'saga',
        penalty = 'elasticnet', 
        l1_ratio = l1_ratio, 
        C = C,
        fit_intercept = True,
        max_iter = 100, 
        n_jobs = n_jobs
    )

    cv = RepeatedKFold(
        n_splits = n_splits, 
        n_repeats = n_repeats, 
        random_state = random_state
    )
    splits_gen = cv.split(phenos)

    predictions = []
    actuals_indeces = []
    for i in range(0, n_repeats+1):
        split = next(splits_gen)
        train_indeces = split[0]
        test_indeces = split[1]

        model.fit(phenos[train_indeces, :], scores[train_indeces])
        coefs = model.coef_

        # Computer Predictions and Summary Stats
        y_hat = model.predict(phenos[test_indeces, :])
        predictions.append(y_hat)
        actuals_indeces.append(test_indeces)
    
    predictions_aggregations[key] = predictions
    actuals_aggregations[key] = actuals_indeces

ikir_hats = []
results = []
for i in range(0, len(predictions_aggregations['f_kir2dl1'])):
    actual_indeces = actuals_aggregations['f_kir2dl1'][i]
    ikir_hat = np.zeros(predictions_aggregations['f_kir2dl1'][i].shape)
    for key in ikir_labels:
        # np.random.shuffle(predictions_aggregations[key][i])
        if key == 'f_kir2dl2_w':
            A = predictions_aggregations['f_kir2dl2_w'][i] & ~predictions_aggregations['f_kir2dl2_s'][i]
            ikir_hat += A*ikir_labels[key]
        else:
            ikir_hat += predictions_aggregations[key][i]*ikir_labels[key]
    
    #ikir_hat = np.random.random_integers(0, 1, size=ikir_hat.shape)

    ikir_hats.append(ikir_hat)

    scores = scores_df['f_kir_score'].values.reshape(-1,1)
    
    #std_scaler = StandardScaler()
    mm_scaler = MinMaxScaler(feature_range=(0, 1))

    #std_scaler = std_scaler.fit(scores)
    #scores = std_scaler.transform(scores)
    mm_scaler.fit(scores)
    scores_ss = scores[actual_indeces]
    #scores_ss = std_scaler.transform(scores_ss)
    scores_ss = mm_scaler.transform(scores_ss)
    y = scores_ss

    y_hat = ikir_hat.reshape(-1, 1)
    #y_hat = std_scaler.transform(y_hat)
    y_hat = mm_scaler.transform(y_hat)

    #np.random.shuffle(y_hat)

    neg_mae = -1*mean_absolute_error(y_true=y, y_pred=y_hat)
    results.append(neg_mae)

results = np.array(results)
avg_neg_mae = results.mean()

output = {}
output['data_source'] = source_filename
output['avg_neg_mae'] = avg_neg_mae
output['run_id'] = run_id

output = pd.Series(output)
print(output)
output.to_csv(output_filename)


run_time = time.time() - start_time 
print('run time:', run_time)
print('Complete.')

# Clustering Tips when performing feature selection with random forest regression on  highly collinear dataset
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py