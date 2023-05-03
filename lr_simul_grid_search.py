import os, random, math, time, uuid
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

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
random_state_1 = 42
random_state_2 = 21

impute = True
strategy = 'mean'
normalise = True
standardise = True

hyper_params_grid = dict()
hyper_params_grid['solver'] = ['saga']
hyper_params_grid['penalty'] = ['elasticnet'] 
hyper_params_grid['l1_ratio']= [0.61] 
hyper_params_grid['C'] = [0.25]
hyper_params_grid['fit_intercept'] = [True]
hyper_params_grid['max_iter'] = [100] 
hyper_params_grid['n_jobs'] = [n_jobs]

c_step = 0.05
c_min = 0.05
c_max = 0.9 + c_min 
hyper_params_grid['C'] = np.arange(
    c_min, c_max, c_step
)

l1_ratio_step = 0.05
l1_ratio_min = 0.05
l1_ratio_max = 0.9 + l1_ratio_step
hyper_params_grid['l1_ratio'] = np.arange(
    l1_ratio_min, l1_ratio_max, l1_ratio_step
)

source_filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
feature_weights_filename = 'Analysis/LogisticRegression/lr_feature_weights_{}.csv'.format(date_str)
output_filename = 'Analysis/LogisticRegression/lr_results_w_cv_{}.csv'.format(date_str)

#Read in Subset of Immunophenotypes
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

scores_df = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_df = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
ikir_labels = {'f_kir2dl1':1.0, 'f_kir2dl2_s':1.0, 'f_kir2dl2_w':0.5, 'f_kir2dl3':0.75, 'f_kir3dl1':1.0}

predictions_aggregations = {key:[] for key in ikir_labels}
actuals_aggregations = {key:[] for key in ikir_labels}
coef_counts = {key:[] for key in ikir_labels}
optimal_h_parms = {key:{} for key in ikir_labels}
aggregated_weights = np.zeros((len(phenos_subset),))
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
        normalise = normalise, strategy=strategy
    )

    scores = scores.ravel()

    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state_1
    )

    gsc = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=hyper_params_grid,
        cv=cv, 
        scoring='neg_mean_absolute_error', 
        verbose=0, 
        n_jobs=n_jobs # parallelism, -1 means using all processors
    )

    # Perform Grid Search
    grid_search_results = gsc.fit(phenos, scores) # Regress Immunos on iKIR Score
    h_params = grid_search_results.best_params_
    optimal_h_parms[key] = h_params
    avg_neg_mae = grid_search_results.best_score_

    # Instantiate Model    
    model = LogisticRegression(
        solver = h_params['solver'],
        penalty = h_params['penalty'], 
        l1_ratio = h_params['l1_ratio'], 
        C = h_params['C'],
        fit_intercept = h_params['fit_intercept'],
        max_iter = h_params['max_iter'], 
        n_jobs = h_params['n_jobs']
    )

    splits_gen = cv.split(phenos)

    predictions = []
    actuals_indeces = []
    for i in range(0, n_repeats+1):
        split = next(splits_gen)
        train_indeces = split[0]
        test_indeces = split[1]

        model.fit(phenos[train_indeces, :], scores[train_indeces])
        feature_weights = model.coef_[0]
        aggregated_weights += feature_weights
        non_zero_coeff_count  = np.where(np.absolute(feature_weights) > 0)[0].size
        coef_counts[key].append(non_zero_coeff_count)

        # Computer Predictions and Summary Stats
        y_hat = model.predict(phenos[test_indeces, :])
        predictions.append(y_hat)
        actuals_indeces.append(test_indeces)
    
    predictions_aggregations[key] = predictions
    actuals_aggregations[key] = actuals_indeces

pheno_weights = [[phenos_subset[i], aggregated_weights[i]] for i in range(0, len(aggregated_weights), 1)]
pheno_weights_df = pd.DataFrame(pheno_weights, columns=['labels', 'lr_coef'])
pheno_weights_df['abs(lr_coef)'] = np.abs(pheno_weights_df['lr_coef'])
pheno_weights_df = pheno_weights_df.sort_values(by='abs(lr_coef)', ascending=False)
pheno_weights_df = pheno_weights_df[pheno_weights_df['abs(lr_coef)'] > 0].copy()
print(pheno_weights_df)
labels = list(pheno_weights_df['labels'])
pheno_weights_df.to_csv(feature_weights_filename)

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
    #ikir_hat = np.ones(predictions_aggregations['f_kir2dl1'][i].shape)*2.75

    ikir_hats.append(ikir_hat)

    scores = scores_df['f_kir_score'].values.reshape(-1,1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(scores)
    
    scores = scores[actual_indeces]

    y = scaler.transform(scores)
    y_hat = scaler.transform(ikir_hat.reshape(-1, 1))

    neg_mae = -1*mean_absolute_error(y_true=y, y_pred=y_hat)
    results.append(neg_mae)

results = np.array(results)
avg_neg_mae = results.mean()
run_time = time.time() - start_time 

output = {}
output['data_source'] = source_filename
output['avg_neg_mae'] = avg_neg_mae
output['solver'] = hyper_params_grid['solver']
output['penalty'] = hyper_params_grid['penalty']
output['l1_ratio'] = hyper_params_grid['l1_ratio']
output['C'] = hyper_params_grid['C']
output['fit_intercept'] = hyper_params_grid['fit_intercept']
output['max_iter'] = hyper_params_grid['max_iter']
output['n_jobs'] = hyper_params_grid['n_jobs']
for key in optimal_h_parms:
    new_key = '{}_{}'.format(key, 'h_params')
    output[new_key] = optimal_h_parms[key]
output['coef_counts'] = coef_counts
output['run_time'] = run_time
output['run_id'] = run_id

output = pd.Series(output)
output.to_csv(output_filename)

print(output)
print(coef_counts)


print('run time:', run_time)
print('Complete.')